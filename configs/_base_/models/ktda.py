# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="DistillEncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    teach_backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="base",
        frozen_stages=12,
        img_size=256,
        patch_size=14,
        layer_scale_init_value=1e-5,
        out_indices=(2, 5, 8, 11),
        out_type="featmap",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-base.pth",
            prefix="backbone",
        ),
    ),
    backbone=dict(
        type="mmpretrain.ConvNeXt",
        arch="base",
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/convnext-base.pth",
            prefix="backbone.",
        ),
    ),
    fam=dict(
        type="FAM",
        in_channels=[128, 256, 512, 1024],
        out_channels=768,
        output_size=19,
    ),
    neck=dict(
        type="MultiLevelNeck",
        in_channels=[768, 768, 768, 768],
        out_channels=768,
        scales=[4, 2, 1, 0.5],
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
