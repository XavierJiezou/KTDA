from glob import glob
import argparse
import os
from typing import Tuple, List
from PIL import Image
from rich.progress import track
from vegseg.datasets import GrassDataset


def get_args() -> Tuple[str, str, int]:
    """
    get args
    return:
        --dataset_path: dataset path.
        --output_path: output path for saving.
        --num: num of image to show. -1 means all.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/grass")
    parser.add_argument("--output_path", type=str, default="all_dataset.png")
    parser.add_argument("--num", default=-1, type=int, help="num of image to show")
    args = parser.parse_args()
    return args.dataset_path, args.output_path, args.num


def get_image_and_mask_paths(
    dataset_path: str, num: int
) -> Tuple[List[str], List[str]]:
    """
    get image and mask paths from dataset path.
    return:
        image_paths: list of image paths.
        mask_paths: list of mask paths.
    """
    image_paths = glob(os.path.join(dataset_path, "img_dir", "*", "*.tif"))
    if num != -1:
        image_paths = image_paths[:num]
    mask_paths = [
        filename.replace("tif", "png").replace("img_dir", "ann_dir")
        for filename in image_paths
    ]
    return image_paths, mask_paths


def get_palette() -> List[int]:
    """
    get palette of dataset.
    return:
        palette: list of palette.
    """
    palette = []
    palette_list = GrassDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette


def paste_image_mask(image_path: str, mask_path: str) -> Image.Image:
    """
    paste image and mask together
    Args:
        image_path (str): path to image.
        mask_path (str): path to mask.
    return:
        image_mask: image with mask,is Image.
    """
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert("P")
    palette = get_palette()
    mask.putpalette(palette)
    mask = mask.convert("RGB")
    image_mask = Image.new("RGB", (image.width * 2, image.height))
    image_mask.paste(image, (0, 0))
    image_mask.paste(mask, (image.width, 0))
    return image_mask


def paste_all_images(all_images: List[Image.Image], output_path: str) -> None:
    """
    paste all images together and save it.
    Args:
        all_images (List[Image.Image]): list of image.
        output_path (str): path to save.
    Return:
        None
    """
    widths = [image.width for image in all_images]
    heights = [image.height for image in all_images]
    width = max(widths)
    height = sum(heights)
    all_image = Image.new("RGB", (width, height))
    for i, image in enumerate(all_images):
        all_image.paste(image, (0, sum(heights[:i])))
    all_image.save(output_path)


def main():
    dataset_path, output_path, num = get_args()
    image_paths, mask_paths = get_image_and_mask_paths(dataset_path, num)
    all_images = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image_mask = paste_image_mask(image_path, mask_path)
        all_images.append(image_mask)
    paste_all_images(all_images, output_path)


if __name__ == "__main__":
    # example usage: python tools/dataset_tools/dataset_show.py --dataset_path data/grass --output_path all_dataset.png
    main()
