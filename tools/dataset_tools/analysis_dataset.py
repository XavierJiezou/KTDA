from glob import glob
from typing import Tuple,List
import os
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

def get_args()->Tuple[str, str]:
    """
    Return:
        --dataset_dir: dataset dir.
        --save_dir: save dir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='data/grass')
    parser.add_argument('--save_dir', type=str, default='dataset_num_analysis.png')
    args = parser.parse_args()
    return args.dataset_dir, args.save_dir

def get_mask_files(dataset_dir: str)->List[str]:
    """
    get mask files from dataset dir.
    Args:
        dataset_dir: dataset dir.
    Return:
        mask_filenames: list of mask filenames.
    """
    mask_filenames = glob(os.path.join(dataset_dir, "ann_dir", "*", "*.png"))
    return mask_filenames

def main():
    dataset_dir, save_dir = get_args()
    mask_filenames = get_mask_files(dataset_dir)
    statistic = {}
    for mask_filename in mask_filenames:
        mask = np.array(Image.open(mask_filename))
        classes = np.unique(mask)
        for class_ in classes:
            class_ = int(class_)
            if class_ not in statistic:
                statistic[class_] = 0
            statistic[(class_)] += int(np.sum(mask == class_))
    
    classes = list(statistic.keys())
    clasees_num = list(statistic.values())

    plt.title("Dataset Analysis")
    bars = plt.bar(classes, clasees_num)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 5, str(height), ha='center', va='bottom')
    plt.savefig(save_dir,dpi=300)
    

if __name__ == "__main__":
    main()