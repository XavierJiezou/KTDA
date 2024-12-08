from mmseg.apis import init_model,inference_model
from PIL import Image
from vegseg.datasets import GrassDataset
import numpy as np
from typing import Tuple, List
from glob import glob
import torch
from mmeval import MeanIoU
from vegseg.models import DistillEncoderDecoder

def get_iou(pred: np.ndarray, gt: np.ndarray, num_classes=5):
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
    palette_list = GrassDataset.METAINFO["palette"]
    for palette_item in palette_list:
        palette.extend(palette_item)
    return palette

config = "work_dirs/experiment_p/experiment_p.py"
checkpoint = "work_dirs/experiment_p/best_mIoU_iter_22770.pth"
device = "cuda:4"
image_paths = glob("data/grass/img_dir/*/*.tif")
batch_size = 64
mask_paths = [filename.replace("img_dir","ann_dir").replace(".tif",".png") for filename in image_paths]
model: DistillEncoderDecoder = init_model(config=config, checkpoint=checkpoint, device=device)
model.decode_head = model.auxiliary_head
model.eval()

save_mask = None
cur_iou = 0
save_filename = None


for i in range(0,len(image_paths),batch_size):
    end_index = min(len(image_paths),i+batch_size)
    image_paths_list = image_paths[i:end_index]
    mask_paths_list = mask_paths[i:end_index]
    results = inference_model(model, image_paths_list)
    for mask_path,result in zip(mask_paths_list,results):

        mask = np.array(Image.open(mask_path))
        pred = result.pred_sem_seg.data.cpu().numpy()[0].astype(np.uint8)
        iou = get_iou(pred,mask)
        if iou > cur_iou and len(np.unique(mask).reshape(-1)) > 3:
            cur_iou = iou
            save_mask = pred
            save_filename = mask_path


auxiliary_img = Image.fromarray(save_mask).convert('P')
palette = get_palette()
auxiliary_img.putpalette(palette)
auxiliary_img.save("auxiliary.png")
print(save_filename)
image_filename = save_filename.replace(".png",".tif").replace("ann_dir","img_dir")
Image.open(image_filename).save("image.png")
Image.open(save_filename).save("mask.png")

