_base_ = [
    '../_base_/reg_default_runtime.py',
    '../_base_/models/base_transmorph.py',
    '../_base_/schedules/default_schedule.py',
    '../_base_/datasets/ixi_atlas-to-scan_160x192x224.py',
]

experiment_name = 'transmorph-bspl_ixi_atlas-to-scan'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

img_size = (160, 192, 224)
window_size = (5, 6, 7)
cps = (3, 3, 3)
flow_dim = 48

model = dict(
    backbone=dict(
        type='TransMorphBSpline',
        window_size=window_size,
        flow_dim=flow_dim,
    ),
    flow=[
        dict(
            type='ResizeFlow',
            img_size=img_size,
            in_channels=flow_dim,
            resize_channels=(32, 32),
            cps=cps),
        dict(
            type='BSplineTransform',
            img_size=img_size,
            cps=cps,
            nsteps=7,
            svf=True)
    ],
    head=dict(
        type='BSplineRegistrationHead',
        img_size=img_size,
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='GradLoss', penalty='l2'),
        normalization=False),
)
