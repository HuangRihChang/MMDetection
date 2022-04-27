_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            _delete_=True,
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            in_channels=256,
            conv_out_channels=1024,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0))))

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=10000)

checkpoint_config = dict(interval=1000)

data = dict(
    samples_per_gpu=3)

optimizer_config = dict(
 _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

log_config = dict(interval=1) # print log every inter (included loss value)

evaluation = dict(interval=1000, metric='bbox', save_best='bbox_mAP') # evaluate every 100 iters
