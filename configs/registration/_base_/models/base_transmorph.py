model = dict(
    type='BaseRegister',
    data_preprocessor=dict(type='RegDataPreprocessor'),
    backbone=dict(
        type='TransMorph',
        window_size=None,  # set by user
        flow_dim=None,  # set by user
    ),
    flow=dict(
        type='DefaultFlow',
        in_channels=None,  # set by user
    ),
    head=dict(
        type='DeformableRegistrationHead',
        img_size=None,  # set by user
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='GradLoss', penalty='l2')),
)
