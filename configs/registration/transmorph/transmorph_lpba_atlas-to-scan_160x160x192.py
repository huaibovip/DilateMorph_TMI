_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule_1k.py',
    '../_base_/datasets/lpba_atlas-to-scan_160x160x192.py',
]

experiment_name = 'transmorph_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

flow_dim = 16
img_size = (160, 160, 192)

model = dict(
    backbone=dict(window_size=(5, 5, 6), flow_dim=flow_dim),
    flow=dict(in_channels=flow_dim),
    head=dict(img_size=img_size),
)
