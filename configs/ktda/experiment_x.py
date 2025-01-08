_base_ = [
    "../_base_/models/ktda.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    backbone=dict(
        _delete_=True,
        type="mmpretrain.TinyViT",
        arch="5m",
        img_size=(256, 256),
        window_size=[7, 7, 14, 7],
        out_indices=(0, 1, 2, 3),
        drop_path_rate=0.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/tiny_vit_5m_imagenet.pth",
            prefix="backbone",
        ),
    ),
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    fmm=dict(type="FMM", in_channels=[768, 768, 768, 768]),
    fam=dict(
        in_channels=[128, 160, 320, 320],
    ),
)
