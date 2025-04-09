# scan_to_scan

dataset_type = 'OASISRegistrationDataset'
data_root = 'data/reg/OASIS_L2R_2021_task03'

train_pipeline = [
    dict(
        type='LoadBundleVolumeFromFile',
        keys=['source', 'target'],
        return_seg=True),
    # NOTE: users should implement their own keyMapper and Pack operation
    dict(type='InjectMeta', meta=dict(num_classes=36)),
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
        type='LoadQuadVolumeFromFile',
        keys=['source', 'target'],
        return_seg=True),
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
    dict(type='InjectMeta', meta=dict(num_classes=36, interp='nearest')),
    dict(type='PackInputs', keys=['source_img', 'target_img'])
]

test_pipeline = [
    dict(
        type='LoadQuadVolumeFromFile',
        keys=['source', 'target'],
        return_seg=True),
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
    dict(type='InjectMeta', meta=dict(num_classes=36, interp='bilinear')),
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
        data_prefix=dict(source='All'),
        filename_tmpl=dict(source='{}'),
        search_key='source',
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
        data_prefix=dict(data='Test'),
        filename_tmpl=dict(data='{}'),
        search_key='data',
    ))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    # persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        data_prefix=dict(data='Test'),
        filename_tmpl=dict(data='{}'),
        search_key='data',
    ))

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
        # output_dir='./work_dirs/oasis3d/submission/task_03',
        format_only=False,
        save_metric=True),
    dict(type='JacobianMetric', metrics=['npj', 'sdlogj'], output_dir=None),
    dict(type='SurfaceDistanceMetric', ignore_index=0, output_dir=None),
]
