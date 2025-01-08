_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/water.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/water_schedule.py",
]

data_preprocessor = dict(size=(512, 512))
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint=checkpoint)),
    decode_head=dict(num_classes=6),
)
