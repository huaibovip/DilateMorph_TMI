_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/oasis_scan-to-scan_160x192x224.py',
]

experiment_name = 'transmorph_oasis_scan-to-scan_swap_supervised'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)
flow_dim = 16

model = dict(
    backbone=dict(
        type='TransMorphHalf',
        window_size=(5, 6, 7),
        flow_dim=flow_dim,
    ),
    flow=[
        dict(type='DefaultFlow', in_channels=flow_dim),
        dict(
            type='UpsampleFlow',
            scale_factor=2,
            mode='trilinear',
            align_corners=False)
    ],
    head=dict(img_size=img_size, loss_seg=dict(type='DiceLoss')),
)

# for exchnage data in single iter
train_cfg = dict(
    type='ExchangeEpochBasedTrainLoop',
    with_seg=True,
    max_epochs=500,
    val_begin=1,
    val_interval=1,
)
