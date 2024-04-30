_base_ = './faster-rcnn_hrnetv2p-w40-1x_coco.py'

# learning policy
max_epochs = 300
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]

default_hooks = dict(
    # Save weights every 10 epochs and a maximum of three weights can be saved.
    # The best model is saved automatically during model evaluation
    checkpoint=dict(interval=10, max_keep_ckpts=3, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))


visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])