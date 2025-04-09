_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/oasis_scan-to-scan_160x192x224.py',
]

experiment_name = 'transmorph_oasis_scan-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)
flow_dim = 16

model = dict(
    backbone=dict(window_size=(5, 6, 7), flow_dim=flow_dim),
    flow=dict(type='DefaultFlow', in_channels=flow_dim),
    head=dict(img_size=img_size),
)

param_scheduler = None
default_hooks = dict(checkpoint=dict(save_param_scheduler=False))
