_base_ = 'The yolov7-l base file path'

data_root = 'Dataset root'
class_name = ('car', 'person', 'cyclist', 'truck', 'van')
num_classes = len(class_name)
metainfo = dict(classes=class_name)
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
max_epochs = 100
train_batch_size_per_gpu = 4
train_num_workers = 4

load_from = 'Pre-trained weighted link on COCO'
model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        prior_generator=dict(base_sizes=anchors)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train annotation file path',
        data_prefix=dict(img='train images path')))

val_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val annotation file path',
        data_prefix=dict(img='val images path')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

val_evaluator = dict(ann_file=data_root + 'val annotation file path')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])