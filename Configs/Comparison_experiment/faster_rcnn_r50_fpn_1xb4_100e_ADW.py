_base_ = 'The base faster-rcnn path'

model = dict(roi_head=dict(bbox_head=dict(num_classes=5)))
metainfo = {
    'classes': ('car', 'person', 'cyclist', 'truck', 'van')
}

train_cfg = dict(max_epochs=100, val_interval=10)
data_root = 'Dataset root'
train_dataloader=dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        ann_file='train annotations files root',
        data_prefix=dict(img='train images root'),
        metainfo=metainfo,
        pipeline={{_base_.train_pipeline}}
    )
)
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        ann_file='val annotations files root',
        data_prefix=dict(img='val images root'),
        metainfo=metainfo,
        pipeline={{_base_.test_pipeline}}
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(ann_file=data_root + 'val annotations files root')
test_evaluator = val_evaluator
_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')]
default_hooks=dict(
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=2, save_best='auto'),
    logger=dict(type='LoggerHook', interval=12)
)
load_from='Pre-trained weighted link on COCO'