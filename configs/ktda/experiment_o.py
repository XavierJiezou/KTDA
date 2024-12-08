_base_ = [
    "../_base_/models/ktda.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    fmm=dict(
        type="fmm",
        in_channels=[768, 768, 768, 768],
        model_type="vitBlock",
        mlp_nums=3,
    ),
)
