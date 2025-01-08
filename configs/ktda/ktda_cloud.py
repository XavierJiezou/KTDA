_base_ = [
    "../_base_/models/ktda.py",
    "../_base_/datasets/l8_biome.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/other_dataset_scedule.py",
]

data_preprocessor = dict(size=(512, 512))
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4),
    fam=dict(output_size=37),
    fmm=dict(
        type="FMM",
        in_channels=[768, 768, 768, 768],
        model_type="vitBlock",
        mlp_nums=4,
    ),
)
train_dataloader = dict(batch_size=2,num_workers=2)
val_dataloader = dict(batch_size=2,num_workers=2)
test_dataloader = dict(batch_size=2,num_workers=2)
