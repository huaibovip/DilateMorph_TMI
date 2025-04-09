default_scope = 'mmipt'
save_dir = './work_dirs'
load_from = None
resume = False

log_level = 'INFO'
log_processor = dict(
    type='LogProcessor',
    window_size=100,
    log_with_hierarchy=True,
    by_epoch=True)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'))

default_hooks = dict(
    timer=dict(type='mmengine.IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        out_dir=save_dir,
        by_epoch=True,
        max_keep_ckpts=1,  # TODO
        save_best='mDice',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook'))
# custom_hooks = [dict(type='VisualizationHook')]

visualizer = dict(
    type='RegVisualizer',
    vis_backends=[
        # dict(type='LocalVisBackend'),
        # dict(type='WandbVisBackend'),
        dict(type='TensorboardVisBackend')
    ],
    img_keys=[
        'pred_flow',
        'pred_grid',
        'pred_seg',
        'pred_img',
        'target_img',
        'target_seg',
    ])

# adding randomness setting
# randomness=dict(seed=0)
