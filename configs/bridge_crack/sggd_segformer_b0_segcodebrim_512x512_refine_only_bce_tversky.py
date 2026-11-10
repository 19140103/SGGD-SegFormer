_base_ = './segformer_b0_segcodebrim_512x512_binary_bce_tversky.py'

model = dict(
    decode_head=dict(
        type='StructureGuidedGatedDecoderHead',
        enable_prior=False,
        enable_refinement=True,
        refinement_reduction=4))

work_dir = 'work_dirs/sggd_segformer_b0_segcodebrim_512x512_refine_only_bce_tversky'


