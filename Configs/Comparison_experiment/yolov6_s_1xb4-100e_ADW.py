_base_ = 'The yolov6-s base file path'

data_root = 'Dataset root'
class_name = ('car', 'person', 'cyclist', 'truck', 'van')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

max_epochs = 100
train_batch_size_per_gpu = 4
train_num_workers = 4
num_last_epochs = 5

load_from = 'Pre-trained weighted link on COCO'  

model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(
        initial_assigner=dict(num_classes=num_classes),
        assigner=dict(num_classes=num_classes)))

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

val_evaluator = dict(ann_file=data_root + 'annotations/val_label.json')
test_evaluator = val_evaluator

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - num_last_epochs

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)])
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackent(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])