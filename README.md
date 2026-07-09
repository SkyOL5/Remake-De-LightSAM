# Working | Leveraging Modality-Specific Priors: A Decoupled SAM Framework with Guided Attention for Medical Image Segmentation

## 📰News

**[2026.07.09]** Model Structure Fixed. Do more experiments.


## 🛠Setup

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install albumentations==0.5.2

```

**Note**: Please refer to requirements.txt


## 📚Data Preparation

The structure is as follows.
```
De-LightSAM
├── datasets
│   ├── image_1024
│     ├── ISIC_0000000.png
|     ├── ...
|   ├── mask_1024
│     ├── ISIC_0000000.png
|     ├── ...
```

## 🎪Segmentation Model Zoo
We provide all pre-trained models here.
| MA-Backbone | MC | Checkpoints |
|-----|------|-----|
|TinyViT| Dermoscopy | [Link](https://drive.google.com/file/d/1kikT1Sjp6TBJQqBJgM80dP2nSTf2PpJM/view?usp=sharing)|


## 📜Citation
If you find this work helpful for your project, please consider citing the following paper:
```

```
