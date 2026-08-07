_base_ = './sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py'

load_from = None  # Set to a SegCODEBRIM pretrained checkpoint for target-domain fine-tuning.

data_root = 'data/dacl10k_crack_fewshot10_mmseg'
train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train')))
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='img_dir/val', seg_map_path='ann_dir/val')))
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        data_root=data_root,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(
            img_path='img_dir/test', seg_map_path='ann_dir/test')))

optim_wrapper = dict(
    optimizer=dict(lr=1e-5),
    accumulative_counts=4,
    clip_grad=dict(max_norm=1.0, norm_type=2))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=200,
        end=3000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=3000, val_interval=500)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=2))
work_dir = 'work_dirs/sggd_segformer_b0_dacl10k_fewshot10_512x512_bce_tversky'


