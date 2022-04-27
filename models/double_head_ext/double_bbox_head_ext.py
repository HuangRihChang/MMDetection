# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import HEADS
from mmdet.models.losses import accuracy
from mmdet.models.roi_heads.bbox_heads import DoubleConvFCBBoxHead
from mmdet.core import multiclass_nms

from mmcv.runner import force_fp32


@HEADS.register_module()
class DoubleConvFCBBoxHeadExt(DoubleConvFCBBoxHead):
    def __init__(self,
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHeadExt, self).__init__(**kwargs)

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.fc_reg_from_fc_head = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls_from_conv_head = nn.Linear(self.fc_out_channels, self.num_classes + 1)

    def forward(self, x_cls, x_reg):
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        conv_bbox_pred = self.fc_reg(x_conv) # reg_from_conv_head
        conv_cls_score = self.fc_cls_from_conv_head(x_conv) # cls_from_conv_head

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        fc_cls_score = self.fc_cls(x_fc) # cls_from_fc_head
        fc_bbox_pred = self.fc_reg_from_fc_head(x_fc) # reg_from_fc_head

        return fc_cls_score, fc_bbox_pred, conv_cls_score, conv_bbox_pred

    def loss(self,
             fc_cls_score,
             fc_bbox_pred,
             conv_cls_score,
             conv_bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):

        lamda_loss_fc = 0.7 # static number same as paper
        lamda_loss_conv = 0.8 # static number same as paper

        losses = dict()
        avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
        if (fc_cls_score is not None) and (fc_bbox_pred is not None):
            if fc_cls_score.numel() > 0:
                loss_cls_from_fc = self.loss_cls(
                    fc_cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)

            loss_bbox_pred_from_fc = self.reuse_loss_bbox(
                fc_bbox_pred,
                rois,
                labels,
                bbox_targets,
                bbox_weights,
                reduction_override
            )

            loss_fc = lamda_loss_fc * loss_cls_from_fc + (1 - lamda_loss_fc) * loss_bbox_pred_from_fc
                
            if isinstance(loss_fc, dict):
                losses.update(loss_fc)
            else:
                losses['loss_of_fc'] = loss_fc
            if self.custom_activation:
                acc_ = self.loss_cls.get_accuracy(fc_cls_score, labels)
                losses.update(acc_)
            else:
                losses['acc'] = accuracy(fc_cls_score, labels)
        if (conv_cls_score is not None) and (conv_bbox_pred is not None):
            if conv_cls_score.numel() > 0:
                loss_cls_from_conv = self.loss_cls(
                    conv_cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                loss_bbox_pred_from_conv = self.reuse_loss_bbox(conv_bbox_pred, rois, labels, bbox_targets, bbox_weights, reduction_override)
            losses['loss_of_conv'] = (1 - lamda_loss_conv) * loss_cls_from_conv + lamda_loss_fc * loss_bbox_pred_from_conv
        return losses

    def reuse_loss_bbox(self, bbox_pred, rois, labels, bbox_targets, bbox_weights, reduction_override=None):
        bg_class_ind = self.num_classes
        # 0~self.num_classes-1 are FG, self.num_classes is BG
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        # do not perform bounding box regression for BG anymore.
        if pos_inds.any():
            if self.reg_decoded_bbox:
                # When the regression loss (e.g. `IouLoss`,
                # `GIouLoss`, `DIouLoss`) is applied directly on
                # the decoded bounding boxes, it decodes the
                # already encoded coordinates to absolute format.
                bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
            else:
                pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), -1,
                    4)[pos_inds.type(torch.bool),
                    labels[pos_inds.type(torch.bool)]]
            return self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds.type(torch.bool)],
                bbox_weights[pos_inds.type(torch.bool)],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        else:
            return bbox_pred[pos_inds].sum()

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   fc_cls_score,
                   conv_cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed. Has shape (num_boxes, 5).
                last dimension 5 arrange as (batch_index, x1, y1, x2, y2).
            cls_score (Tensor): Box scores, has shape
                (num_boxes, num_classes + 1).
            bbox_pred (Tensor, optional): Box energies / deltas.
                has shape (num_boxes, num_classes * 4).
            img_shape (Sequence[int], optional): Maximum bounds for boxes,
                specifies (H, W, C) or (H, W).
            scale_factor (ndarray): Scale factor of the
               image arrange as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]:
                First tensor is `det_bboxes`, has the shape
                (num_boxes, 5) and last
                dimension 5 represent (tl_x, tl_y, br_x, br_y, score).
                Second tensor is the labels with shape (num_boxes, ).
        """

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            fc_scores = self.loss_cls.get_activation(fc_cls_score)
            conv_scores = self.loss_cls.get_activation(conv_cls_score)
        else:
            fc_scores = F.softmax(
                fc_cls_score, dim=-1) if fc_cls_score is not None else None
            conv_scores = F.softmax(
                conv_cls_score, dim=-1) if conv_cls_score is not None else None

        if self.training:
            scores = fc_scores
        else:
            # Complementary Fusion of Classifiers
            scores = fc_scores + conv_scores * (1 - fc_scores)

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels