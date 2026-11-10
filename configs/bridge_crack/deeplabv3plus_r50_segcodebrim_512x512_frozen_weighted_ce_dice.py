_base_ = './deeplabv3plus_r50_segcodebrim_512x512_frozen_ce_dice.py'

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=[1.0, 20.0],
                loss_name='loss_weighted_ce',
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_name='loss_dice',
                loss_weight=3.0)
        ]),
    auxiliary_head=dict(
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=[1.0, 20.0],
            loss_name='loss_aux_weighted_ce',
            loss_weight=0.4)))

work_dir = 'work_dirs/deeplabv3plus_r50_segcodebrim_512x512_frozen_weighted_ce_dice'

# DeepLabV3+ contains an image-level ASPP pooling branch whose BatchNorm receives
# a 1x1 feature map. A per-GPU batch size of 2 avoids the single-sample BN error
# while keeping memory use manageable on the RTX 3080.
train_dataloader = dict(batch_size=2, num_workers=0, persistent_workers=False)


