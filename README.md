<div align="center">

# KTDA

Knowledge Transfer and Domain Adaptation for Fine-Grained Remote Sensing Image Segmentation

[![arXiv Paper](https://img.shields.io/badge/arXiv-2412.06664-B31B1B)](https://arxiv.org/abs/2412.06664)
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

- Option 2: Use Pre-configured Environment  

We provide a pre-configured environment (`env.tar.gz`) hosted on Hugging Face. You can download it directly from [Hugging Face](https://huggingface.co/XavierJiezou/ktda-models/blob/main/env.tar.gz). Follow the instructions on the page to set up and activate the environment. 

Once download env.tar.gz, you can extract it using the following command:  

```bash
tar -xzf env.tar.gz -C envs
source envs/bin/activate
conda-unpack
```

## Prepare Data  

We have open-sourced all datasets used in the paper, which are hosted on [Hugging Face Datasets](https://huggingface.co/datasets/XavierJiezou/ktda-datasets). Please follow the instructions on the dataset page to download the data.  

After downloading, organize the dataset as follows:  

```  
KTDA
├── ...
├── data
│   ├── grass
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── l8_biome (cloud)
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

### Step 1: Modify the Configuration File

After downloading the vision transformer models from [Hugging Face](https://huggingface.co/XavierJiezou/ktda-models), make sure to correctly specify the path to the configuration file within your config settings.

For example: 

```python
# configs/_base_/models/ktda.py
model = dict(
    backbone=dict(
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2-base.pth", # you can set vision transformer models path here
        ),
    ),
   
)
```

Update the `configs` directory with your training configuration, or use one of the provided example configurations. You can customize the backbone, dataset paths, and hyperparameters in the configuration file (e.g., `configs/ktda/ktda_cloud.py`).  

### Step 2: Start Training  

Use the following command to begin training on grass dataset:  

```bash  
python tools/train.py configs/ktda/ktda_grass.py
```  

and you can also train on cloud dataset:

```bash  
python tools/train.py configs/ktda/ktda_cloud.py
``` 

### Step 3: Resume or Fine-tune  

To resume training from a checkpoint or fine-tune using pretrained weights, run:  

```bash  
python tools/train.py configs/ktda/ktda_grass.py --resume-from path/to/checkpoint.pth  
```

## Evaluation

All model weights used in the paper have been open-sourced and are available on [Hugging Face Models](https://huggingface.co/XavierJiezou/ktda-models).

Use the following command to evaluate the trained model:  

```bash  
python tools/test.py configs/ktda/ktda_grass.py path/to/checkpoint.pth  
```

Alternatively, you can find the evaluation results in the [eval_result](eval_result) folder within this repository.

## Visualization

We have uploaded the visualization results of various models on the Grass and Cloud datasets to Hugging Face. You can view them at the following link:

[Hugging Face Visualization Results](https://huggingface.co/XavierJiezou/ktda-models/tree/main/visualization)


## Citation

If you use our code or models in your research, please cite with:

```latex
@misc{ktda,
      title={Knowledge Transfer and Domain Adaptation for Fine-Grained Remote Sensing Image Segmentation}, 
      author={Shun Zhang and Xuechao Zou and Kai Li and Congyan Lang and Shiying Wang and Pin Tao and Tengfei Cao},
      year={2024},
      eprint={2412.06664},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.06664}, 
}
```

## Acknowledgments

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=open-mmlab&repo=mmsegmentation)]([https://github.com/python-poetry/poetry](https://github.com/open-mmlab/mmsegmentation))
