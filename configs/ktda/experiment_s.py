_base_ = [
    "../_base_/models/convnextv2_femto_vit_segformer_vegseg.py",
    "../_base_/datasets/grass.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/grass_schedule.py",
]

data_preprocessor = dict(size=(256, 256))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=5),
    auxiliary_head=dict(num_classes=5),
    fmm=dict(type="FMM", in_channels=[768, 768, 768, 768]),
)
