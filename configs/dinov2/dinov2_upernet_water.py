_base_ = [
    "../_base_/models/dinov2_upernet.py",
    "../_base_/datasets/water.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/water_schedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=6),
    auxiliary_head=dict(num_classes=6)
)
