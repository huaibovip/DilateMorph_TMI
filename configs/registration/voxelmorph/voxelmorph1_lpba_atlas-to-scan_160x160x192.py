_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_voxelmorph.py',
    '../_base_/schedules/default_schedule_1k.py',
    '../_base_/datasets/lpba_atlas-to-scan_160x160x192.py',
]

experiment_name = 'voxelmorph1_lpba_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 160, 192)
nb_features = [
    [8, 32, 32, 32],  # encoder
    [32, 32, 32, 32, 32, 8, 8]  # decoder
]

model = dict(
    backbone=dict(img_size=img_size, nb_unet_features=nb_features),
    flow=dict(in_channels=nb_features[-1][-1]),
    head=dict(img_size=img_size),
)
