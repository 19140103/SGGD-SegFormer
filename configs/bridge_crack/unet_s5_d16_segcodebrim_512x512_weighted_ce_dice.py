_base_ = './unet_s5_d16_segcodebrim_512x512_ce_dice.py'

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

work_dir = 'work_dirs/unet_s5_d16_segcodebrim_512x512_weighted_ce_dice'


