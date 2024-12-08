_base_ = [
    "../_base_/models/dinov2_upernet.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=5,
        loss_decode=[
            dict(type="FocalLoss", use_sigmoid=True, loss_weight=1.0),
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ],
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/head.pth",
            prefix="decode_head",
        ),
    ),
    auxiliary_head=dict(
        num_classes=5,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="work_dirs/dinov2_b_frozen-simpleAdapter/auxiliary_head.pth",
            prefix="auxiliary_head",
        ),
    ),
)
