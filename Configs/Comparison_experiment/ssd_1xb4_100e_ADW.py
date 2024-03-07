_base_ = 'The base SSD path'

model = dict(
    bbox_head=dict(num_classes=5)
)
metainfo = {
    'classes': ('car', 'person', 'cyclist', 'truck', 'van')
}
train_cfg = dict(max_epochs=100, val_interval=10)
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root='Dataset root',
        metainfo=metainfo,
        ann_file='train annotations files root',
        pipeline={{_base_.train_pipeline}},
        data_prefix=dict(img='train images root'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        backend_args={{_base_.backend_args}}
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='Dataset root',
        data_root='data/ADW_dataset/',
        metainfo=metainfo,
        ann_file='val annotations files root',
        data_prefix=dict(img='val images root'),
        pipeline={{_base_.test_pipeline}}
    )
)
val_evaluator = dict(ann_file='val annotations files root')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
custom_hooks = [
    dict(type='CheckInvalidLossHook', interval=12, priority='VERY_LOW')
]
default_hooks=dict(
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=12)
)
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]
load_from = 'Pre-trained weighted link on COCO'