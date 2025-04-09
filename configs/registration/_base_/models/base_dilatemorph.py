model = dict(
    type='BaseRegister',
    data_preprocessor=dict(type='RegDataPreprocessor'),
    backbone=dict(
        type='DilateMorph',
        img_size=None,  # set by user
        use_checkpoint=False,
    ),
    flow=dict(
        type='DefaultFlow',
        in_channels=None,  # set by user
    ),
    head=dict(
        type='DeformableRegistrationHead',
        img_size=None,  # set by user
        loss_sim=dict(type='NCCLoss'),
        loss_reg=dict(type='GradLoss', penalty='l2', loss_weight=3.0)),
)
