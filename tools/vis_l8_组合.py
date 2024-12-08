from glob import glob
from mmeval import MeanIoU
from PIL import Image
import numpy as np
from typing import List
from vegseg.datasets import L8BIOMEDataset
from matplotlib import pyplot as plt
import os

def give_color_to_mask(
    mask: Image.Image | np.ndarray, palette: List[int]
) -> Image.Image:
    """
    Args:
        mask: mask to color, numpy array or PIL Image.
        palette: palette of dataset.
    return:
        mask: mask with color.
    """
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    mask = mask.convert("P")
    mask.putpalette(palette)
    return mask

def get_iou(pred: np.ndarray, gt: np.ndarray, num_classes=2):
    pred = pred[np.newaxis]
    gt = gt[np.newaxis]
    miou = MeanIoU(num_classes=num_classes)
    result = miou(pred, gt)
    return result["mIoU"] * 100


def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = L8BIOMEDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette

def main():
    ktda = glob("data/vis/ktda/*.png")

    all_images = [
        "cdnetv1",
        "cdnetv2",
        "hrcloudnet",
        "input",
        "kappamask",
        "ktda",
        "label",
        "mcdnet",
        "scnn",
        "unetmobv2",
    ]
    model_order = [
        "ktda",
        "cdnetv1",
        "cdnetv2",
        "hrcloudnet",
        "kappamask",
        "mcdnet",
        "scnn",
        "unetmobv2",
    ]
    palette = get_palette()
    for ktda_path in ktda:
        images_paths = [
            ktda_path.replace("ktda", filename) for filename in all_images
        ]
        model_name_mask = {}
        model_iou = {}
        label_path = ktda_path.replace("ktda", "label")
        for image_path in images_paths:
            model_name = image_path.split("/")[-2]
            if model_name in ["input", "label"]:
                continue
            model_name_mask[model_name] = np.array(Image.open(image_path))
            model_iou[model_name] = get_iou(
                model_name_mask[model_name], np.array(Image.open(label_path)),num_classes=4
            )
        result_iou_sorted = sorted(model_iou.items(), key=lambda x: x[1], reverse=True)
        if result_iou_sorted[0][0] != "ktda":
            continue
        input_path = ktda_path.replace("ktda", "input")

        plt.figure(figsize=(32, 8))
        plt.subplots_adjust(wspace=0.01)
        plt.subplot(1, 10, 1)
        plt.imshow(Image.open(input_path))
        plt.axis("off")

        plt.subplot(1, 10, 2)
        plt.imshow(give_color_to_mask(Image.open(label_path), palette=palette))
        plt.axis("off")

        for i, model_name in enumerate(model_order):
            plt.subplot(1, 10, i + 3)
            plt.imshow(give_color_to_mask(model_name_mask[model_name], palette))
            plt.axis("off")
        base_name = os.path.basename(ktda_path).split(".")[0]
        diff_iou = result_iou_sorted[0][1] - result_iou_sorted[1][1]
        plt.savefig(
            f"l8_vis/{diff_iou:.2f}_{base_name}.svg",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()



if __name__ == "__main__":
    main()
