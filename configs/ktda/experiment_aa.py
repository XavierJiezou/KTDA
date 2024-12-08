_base_ = [
    "../_base_/models/convnextv2_femto_vit_segformer_vegseg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    teach_backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="large",
        frozen_stages=24,
        img_size=256,
        patch_size=14,
        layer_scale_init_value=1e-5,
        out_indices=(7, 11, 15, 23),
        out_type="featmap",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-large.pth",
            prefix="backbone",
        ),
    ),
    fam=dict(out_channels=1024),
    decode_head=dict(in_channels=[1024, 1024, 1024, 1024], num_classes=5),
    data_preprocessor=data_preprocessor,
    auxiliary_head=[
        dict(
            type="FCNHead",
            in_channels=1024,
            in_index=i,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=5,
            norm_cfg=dict(type="SyncBN", requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4
            ),
        )
        for i in range(4)
    ],
)
