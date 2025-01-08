_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth"  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
    ),
    decode_head=dict(num_classes=5, in_channels=[64, 128, 320, 512]),
)
