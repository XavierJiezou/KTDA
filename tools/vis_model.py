from glob import glob
import argparse
import os
from typing import Tuple, List
import numpy as np
from mmeval import MeanIoU
from PIL import Image
from matplotlib import pyplot as plt
from mmseg.apis import MMSegInferencer
from ktda.datasets import GrassDataset


def get_iou(pred: np.ndarray, gt: np.ndarray, num_classes=2):
    pred = pred[np.newaxis]
    gt = gt[np.newaxis]
    miou = MeanIoU(num_classes=num_classes)
    result = miou(pred, gt)
    return result["mIoU"] * 100


def get_args() -> Tuple[str, str, int]:
    """
    get args
    return:
        --models: all_models path.
        --device: device to use.
        --dataset_path: dataset path.
        --output_path: output path for saving.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="work_dirs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset_path", type=str, default="data/grass")
    args = parser.parse_args()
    return args.models, args.device, args.dataset_path


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


def get_image_and_mask_paths(
    dataset_path: str, num: int
) -> Tuple[List[str], List[str]]:
    """
    get image and mask paths from dataset path.
    return:
        image_paths: list of image paths.
        mask_paths: list of mask paths.
    """
    image_paths = glob(os.path.join(dataset_path, "img_dir", "val", "*.tif"))
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


def init_all_models(models_path: str, device: str):
    """
    init all models
    Args:
        models_path (str): path to all models.
        device (str): device to use.
    Return:
        models (dict): dict of models.
    """
    models = {}
    all_models = os.listdir(models_path)
    for model_path in all_models:
        model_name = model_path
        model_path = os.path.join(models_path, model_path)
        config_path = glob(os.path.join(model_path, "*.py"))[0]
        weight_path = glob(os.path.join(model_path, "best_mIoU_iter_*.pth"))[0]
        inference = MMSegInferencer(
            config_path,
            weight_path,
            device=device,
            classes=GrassDataset.METAINFO["classes"],
            palette=GrassDataset.METAINFO["palette"],
        )
        models[model_name] = inference
    return models


def main():
    models_path, device, dataset_path = get_args()
    image_paths, mask_paths = get_image_and_mask_paths(dataset_path, -1)
    palette = get_palette()
    models = init_all_models(models_path, device)
    os.makedirs("vis_results", exist_ok=True)
    for image_path, mask_path in zip(image_paths, mask_paths):
        result_eval = {}
        result_iou = {}
        mask = Image.open(mask_path)
        for model_name, inference in models.items():
            predictions: np.ndarray = inference(image_path)["predictions"]
            predictions = predictions.astype(np.uint8)
            result_eval[model_name] = predictions
            result_iou[model_name] = get_iou(predictions, np.array(mask), num_classes=5)

        # 根据iou 进行排序
        result_iou_sorted = sorted(result_iou.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(36, 3))
        plt.subplot(1, len(models) + 2, 1)
        plt.imshow(Image.open(image_path))
        plt.axis("off")
        plt.title("Input")

        plt.subplot(1, len(models) + 2, 2)
        plt.imshow(give_color_to_mask(mask, palette=palette))
        plt.axis("off")
        plt.title("Label")

        for i, (model_name, _) in enumerate(result_iou_sorted):
            plt.subplot(1, len(models) + 2, i + 3)
            plt.imshow(give_color_to_mask(result_eval[model_name], palette))
            plt.axis("off")
            plt.title(f"{model_name}: {result_iou[model_name]:.2f}")

        base_name = os.path.basename(image_path).split(".")[0]
        plt.savefig(
            f"vis_results/{base_name}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )


if __name__ == "__main__":
    # example usage: python tools/vis_model.py --models work_dirs --device cuda:0 --dataset_path data/grass
    main()
