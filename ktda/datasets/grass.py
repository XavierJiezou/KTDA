# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class GrassDataset(BaseSegDataset):
    """grass segmentation dataset. The file structure should be.

    .. code-block:: none

        ├── data
        │   ├── grass
        │   │   ├── img_dir
        │   │   │   ├── train
        │   │   │   │   ├──0.tif
        │   │   │   │   ├──...
        │   │   │   ├── val
        │   │   │   │   ├──9.tif
        │   │   │   │   ├──...
        │   │   ├── ann_dir
        │   │   │   ├── train
        │   │   │   │   ├──0.png
        │   │   │   │   ├──...
        │   │   │   ├── val
        │   │   │   │   ├──9.png
        │   │   │   │   ├──...
    """

    METAINFO = dict(
        classes=("low", "middle-low", "middle", "middle-high", "high"),
        palette=[
            [185, 101, 71],
            [248, 202, 155],
            [211, 232, 158],
            [138, 191, 104],
            [92, 144, 77],
        ],
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)