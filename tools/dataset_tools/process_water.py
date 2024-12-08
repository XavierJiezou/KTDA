import shutil
from glob import glob
import os
import argparse
import numpy as np
from rich.progress import track
from PIL import Image
from typing import List
from vegseg.datasets import WaterDataset
from sklearn.model_selection import train_test_split


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--raw_path", type=str)
    parse.add_argument("--tmp_dir", type=str)
    parse.add_argument("--save_path", type=str)
    args = parse.parse_args()
    return args.raw_path, args.tmp_dir, args.save_path


def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = WaterDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette


def create_dataset(image_list, ann_list, image_dir, ann_dir, description="Working..."):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    for image_path, ann_path in track(
        zip(image_list, ann_list), total=len(image_list), description=description
    ):
        base_name = os.path.basename(image_path)

        new_image_path = os.path.join(image_dir, base_name)
        new_ann_path = os.path.join(ann_dir, base_name)

        shutil.move(image_path, new_image_path)
        shutil.move(ann_path, new_ann_path)

        mask = Image.open(new_ann_path).convert("P")
        palette = get_palette()
        mask.putpalette(palette)
        mask.save(new_ann_path)


def main():
    classes_mapping = {
        "CDUWD-1": 1,
        "CDUWD-2": 2,
        "CDUWD-3": 3,
        "CDUWD-4": 4,
        "CDUWD-5": 5,
        "CDUWD-6": 0,
    }

    raw_path, tmp_dir, save_path = get_args()

    all_images = glob(os.path.join(raw_path, "*", "images", "*.png"))

    all_labels = [image_path.replace("images", "labels") for image_path in all_images]

    target_image_dir = os.path.join(tmp_dir, "images")
    target_label_dir = os.path.join(tmp_dir, "labels")

    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)

    for image_path, label_path in track(
        zip(all_images, all_labels), total=len(all_images), description="fuse dataset"
    ):
        exists_images = glob(os.path.join(target_image_dir, "*.png"))

        base_name = os.path.basename(image_path)
        if image_path not in exists_images:
            mask = np.array(Image.open(label_path))

            assert list(np.unique(mask)) in [
                [0],
                [1],
                [0, 1],
                [1, 0],
            ], f"The mask image is not binary (it should only contain 0s and 1s),actually is {set(np.unique(mask))}"

            classes_str = image_path.split(os.path.sep)[-3]
            classes = classes_mapping[classes_str]
            mask = np.where(mask == 1, classes, mask)

            # print(classes_str)

            mask = Image.fromarray(mask)
            mask.save(os.path.join(target_label_dir, base_name))
            shutil.copy(image_path, os.path.join(target_image_dir, base_name))
        else:

            exists_label_path = os.path.join(target_label_dir, base_name)
            exists_mask = np.array(Image.open(exists_label_path))

            mask = np.array(Image.open(label_path))
            assert list(np.unique(mask)) in [
                [0],
                [1],
                [0, 1],
                [1, 0],
            ], f"The mask image is not binary (it should only contain 0s and 1s),actually is {set(np.unique(mask))}"
            classes_str = image_path.split(os.path.sep)[-3]
            classes = classes_mapping[classes_str]

            exists_mask = np.where(mask == 1, classes, exists_mask)

            exists_mask = Image.fromarray(exists_mask)
            exists_mask.save(exists_label_path)

    exists_images = glob(os.path.join(target_image_dir, "*.png"))

    exists_labels = [
        image_path.replace("images", "labels") for image_path in exists_images
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        exists_images, exists_labels, test_size=0.2, random_state=42, shuffle=True
    )

    create_dataset(
        X_train,
        y_train,
        os.path.join(save_path, "img_dir", "train"),
        os.path.join(save_path, "ann_dir", "train"),
        description="train dataset",
    )
    create_dataset(
        X_test,
        y_test,
        os.path.join(save_path, "img_dir", "val"),
        os.path.join(save_path, "ann_dir", "val"),
        description="val dataset",
    )

    os.rmdir(target_image_dir)
    os.rmdir(target_label_dir)


if __name__ == "__main__":
    # example python tools/dataset_tools/process_water.py --raw_path data/raw_water_dataset/1024 --tmp_dir data/raw_water_dataset/1024/all_dataset --save_path data/water_1024_1024
    main()
