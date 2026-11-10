_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py',
    '../_base_/datasets/segcodebrim_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_name='loss_ce',
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_name='loss_dice',
                loss_weight=3.0)
        ]),
    auxiliary_head=dict(
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_name='loss_aux_ce',
            loss_weight=0.4)),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, weight_decay=0.01),
    accumulative_counts=4,
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=500,
        end=15000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=15000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

train_dataloader = dict(batch_size=2, num_workers=0, persistent_workers=False)
val_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)
test_dataloader = dict(batch_size=1, num_workers=0, persistent_workers=False)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo'))

randomness = dict(seed=20260802, deterministic=False)

work_dir = 'work_dirs/unet_s5_d16_segcodebrim_512x512_ce_dice'


