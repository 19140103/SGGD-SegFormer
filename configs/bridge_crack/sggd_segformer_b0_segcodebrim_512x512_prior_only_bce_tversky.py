_base_ = './segformer_b0_segcodebrim_512x512_binary_bce_tversky.py'

model = dict(
    decode_head=dict(
        type='StructureGuidedGatedDecoderHead',
        prior_channels=64,
        prior_strength=0.5,
        enable_prior=True,
        enable_refinement=False))

work_dir = 'work_dirs/sggd_segformer_b0_segcodebrim_512x512_prior_only_bce_tversky'


