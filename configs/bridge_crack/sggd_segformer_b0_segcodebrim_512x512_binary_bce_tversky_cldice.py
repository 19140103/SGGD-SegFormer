_base_ = './sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky.py'

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                class_weight=[20.0],
                loss_name='loss_bce',
                loss_weight=1.0),
            dict(
                type='BinaryTverskyLoss',
                alpha=0.3,
                beta=0.7,
                loss_name='loss_binary_tversky',
                loss_weight=1.0),
            dict(
                type='SoftClDiceLoss',
                iter_=10,
                loss_name='loss_cldice',
                loss_weight=0.2)
        ]))

work_dir = 'work_dirs/sggd_segformer_b0_segcodebrim_512x512_binary_bce_tversky_cldice'


