_base_ = './sggd_segformer_b0_dacl10k_fewshot10_512x512_bce_tversky.py'

data_root = 'data/dacl10k_crack_fewshot05_mmseg'

train_dataloader = dict(dataset=dict(data_root=data_root))
val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = dict(dataset=dict(data_root=data_root))

work_dir = 'work_dirs/sggd_segformer_b0_dacl10k_fewshot05_512x512_bce_tversky'



