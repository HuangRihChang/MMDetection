_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=10000)

checkpoint_config = dict(interval=20000)

data = dict(
    samples_per_gpu=3)

optimizer_config = dict(
 _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

log_config = dict(interval=1) # print log every inter (included loss value)

evaluation = dict(interval=1000, metric='bbox', save_best='bbox_mAP') # evaluate every 100 iters
