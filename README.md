<div align="center">

![logo](https://github.com/user-attachments/assets/f9351412-d54a-4ac6-9344-d412fe3b3581)

# KTDA

Knowledge Transfer and Domain Adaptation for Fine-Grained Remote Sensing Image Segmentation

[![Project Page](https://img.shields.io/badge/Project%20Page-KTDA-blue)](https://xavierjiezou.github.io/KTDA/)
[![HugginngFace Models](https://img.shields.io/badge/🤗HugginngFace-Models-orange)](https://huggingface.co/XavierJiezou/ktda-models)
[![HugginngFace Datasets](https://img.shields.io/badge/🤗HugginngFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/ktda-datasets)
<!--[![Overleaf](https://img.shields.io/badge/Overleaf-Open-green?logo=Overleaf&style=flat)](https://www.overleaf.com/project/6695fd4634d7fee5d0b838e5)-->

<!--Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!-->

</div>

<!--This repository serves as the official implementation of the paper **"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images"**. It provides a comprehensive pipeline for semantic segmentation, including data preprocessing, model training, evaluation, and deployment, specifically tailored for cloud segmentation tasks in remote sensing imagery.-->

---


## Installation  

1. Clone the Repository  

```bash  
git clone https://github.com/XavierJiezou/KTDA.git
cd KTDA
```  

2. Install Dependencies  

You can either set up the environment manually or use our pre-configured environment for convenience:  

- Option 1: Manual Installation  

Ensure you are using Python 3.8 or higher, then install the required dependencies:  

```bash  
pip install -r requirements.txt  
```  

## Prepare Data  

We have open-sourced all datasets used in the paper, which are hosted on [Hugging Face Datasets](KTDA). Please follow the instructions on the dataset page to download the data.  

After downloading, organize the dataset as follows:  

```  
Cloud-Adapter
├── ...
├── data
│   ├── grass
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── l8_biome
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
```   

## Training

### Step 1: Download and Convert Weights

1. Download pretrained weights of vision foundation models

You can download the pretrained weights from the [hugging face](https://huggingface.co/XavierJiezou/ktda-models). 

### Step 2: Modify the Configuration File

After downloading the backbone network weights, make sure to correctly specify the path to the configuration file within your config settings.

For example: 

```python
# configs/_base_/models/cloud_adapter_dinov2.py
model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-base.pth", # you can set weight path here
        ),
    ),
   
)
```

Update the `configs` directory with your training configuration, or use one of the provided example configurations. You can customize the backbone, dataset paths, and hyperparameters in the configuration file (e.g., `configs/ktda/ktda_cloud.py`).  

### Step 3: Start Training  

Use the following command to begin training on grass dataset:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ktda/ktda_grass.py
```  

and you can also train on cloud dataset:

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/ktda/ktda_cloud.py
``` 

### Step 4: Resume or Fine-tune  

To resume training from a checkpoint or fine-tune using pretrained weights, run:  

```bash  
python tools/train.py configs/ktda/ktda_grass.py --resume-from path/to/checkpoint.pth  
```

## Evaluation

All model weights used in the paper have been open-sourced and are available on [Hugging Face Models](https://huggingface.co/XavierJiezou/ktda-models).

Use the following command to evaluate the trained model:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/ktda/ktda_grass.py path/to/checkpoint.pth  
```  

## Citation

If you use our code or models in your research, please cite with:

```latex
@misc{ktda,
      title={Knowledge Transfer and Domain Adaptation for Fine-Grained Remote Sensing Image Segmentation}, 
      author={Shun Zhang and Xuechao Zou and Shiying Wang and  Kai Li and Congyan Lang and Pin Tao},
      year={2024},
}
```

## Acknowledgments

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=open-mmlab&repo=mmsegmentation)]([https://github.com/python-poetry/poetry](https://github.com/open-mmlab/mmsegmentation))
