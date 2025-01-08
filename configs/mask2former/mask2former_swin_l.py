_base_ = [
    "../_base_/models/mask2former_swin-b.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]
pretrained = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"  # noqa
data_preprocessor = dict(size=(256, 256))
model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=5,
        num_queries=100,
        in_channels=[192, 384, 768, 1536],
        loss_cls=dict(class_weight=[1.0] * 5 + [0.1]),
    ),
)
