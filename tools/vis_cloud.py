from mmseg.apis import MMSegInferencer
from glob import glob
from ktda.datasets import L8BIOMEDataset
import numpy as np
from typing import List
import os
from PIL import Image
from ktda import models

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


def give_color_to_mask(
    mask: Image.Image | np.ndarray, palette: List[int]
) -> Image.Image:
    """
    give color to mask.
    return:
        color_mask: color mask.
    """
    color_mask = Image.fromarray(mask).convert("P")
    color_mask.putpalette(palette)
    return color_mask


def main():
    config_path = "work_dirs/experiment_p_l8/experiment_p_l8.py"
    weight_path = "work_dirs/experiment_p_l8/best_mIoU_iter_20000.pth"
    inference = MMSegInferencer(
        model=config_path,
        weights=weight_path,
        device="cuda:1",
        classes=L8BIOMEDataset.METAINFO["classes"],
        palette=L8BIOMEDataset.METAINFO["palette"],
    )
    images = glob("data/vis/input/*.png")
    palette = get_palette()
    predictions = inference.__call__(images,batch_size=16)["predictions"]
    for image_path, prediction in zip(images, predictions):
        filename = os.path.basename(image_path)
        filename = os.path.join("data/vis/ktda",filename)
        prediction = prediction.astype(np.uint8)
        color_mask = give_color_to_mask(prediction, palette=palette)
        color_mask.save(filename)

if __name__ == "__main__":
    main()
