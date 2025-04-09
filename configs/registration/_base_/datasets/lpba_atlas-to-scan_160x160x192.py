# atlas_to_scan

dataset_type = 'LPBARegistrationDataset'
data_root = 'data/reg/LPBA40'

train_pipeline = [
    dict(
        type='LoadBundleVolumeFromFile',
        keys=['source', 'target'],
        return_seg=False),
    dict(type='RandomFlip', keys=['source_img', 'target_img'], axes=(1, 2, 3)),
    # NOTE: users should implement their own keyMapper and Pack operation
    dict(
        type='KeyMapper',
        mapping=dict(
            source_shape='ori_source_img_shape',
            target_shape='ori_target_img_shape',
        ),
        remapping=dict(
            source_shape='source_shape',
            target_shape='target_shape',
        )),
    dict(type='PackInputs', keys=['source_img', 'target_img'])
]

val_pipeline = [
    dict(
        type='LoadBundleVolumeFromFile',
        keys=['source', 'target'],
        return_seg=True),
    dict(type='LPBASegmentNormalize', keys=['source_seg', 'target_seg']),
    # NOTE: users should implement their own keyMapper and Pack operation
    dict(
        type='KeyMapper',
        mapping=dict(
            source_shape='ori_source_img_shape',
            target_shape='ori_target_img_shape',
        ),
        remapping=dict(
            source_shape='source_shape',
            target_shape='target_shape',
        )),
    dict(type='InjectMeta', meta=dict(num_classes=55, interp='bilinear')),
    dict(type='PackInputs', keys=['source_img', 'target_img'])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        test_mode=False,
        data_prefix=dict(source='', target='Train'),
        filename_tmpl=dict(source='S01', target='{}'),
        search_key='target',
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline,
        test_mode=True,
        data_prefix=dict(source='', target='Test'),
        filename_tmpl=dict(source='S01', target='{}'),
        search_key='target',
    ))

test_dataloader=val_dataloader

val_evaluator = [
    dict(
        type='DiceMetric',
        iou_metrics=['mDice'],
        ignore_index=0,
        output_dir=None,
        save_metric=False),
]

test_evaluator = [
    dict(
        type='DiceMetric',
        iou_metrics=['mDice'],
        ignore_index=0,
        output_dir=None,
        save_metric=True),
    dict(type='JacobianMetric', metrics=['jdet'], output_dir=None),
    dict(type='SurfaceDistanceMetric', ignore_index=0, output_dir=None),
]
