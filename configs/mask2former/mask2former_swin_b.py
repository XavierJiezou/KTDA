_base_ = [
    "../_base_/models/mask2former_swin-b.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5, loss_cls=dict(class_weight=[1.0] * 5 + [0.1])),
)
