# LFIC-DRASC - Light Field Image Compression Neural Network
Shiyu Feng, [Yun Zhang](https://codec.siat.ac.cn/yunzhang/), [Linwei Zhu](https://zhulinweicityu.github.io/), [Sam Kwong](https://scholars.ln.edu.hk/en/persons/sam-tak-wu-kwong) 



![GitHub stars](https://img.shields.io/github/stars/SYSU-Video/LFIC-DRASC?style=social)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
[![Paper](https://img.shields.io/badge/Paper-arxiv'24-b31b1b.svg)](https://arxiv.org/abs/2409.11711)
![License](https://img.shields.io/github/license/SYSU-Video/LFIC-DRASC)
![Last commit](https://img.shields.io/github/last-commit/SYSU-Video/LFIC-DRASC)

## Project Introduction

 LFIC-DRASC is a deep learning model for light field image compression, which maintains spatial consistency through a special network structure, improving compression efficiency and visual quality.

## Environment Configuration

Main dependencies:
```
torch==2.0.1
torchvision==0.15.2
compressai==1.1.5
pytorch-msssim==1.0.0
```

Complete dependencies can be installed with the following command:
```bash
pip install -r requirements.txt
```

## Usage

### Training Model

```bash
python train.py -d dataset --N 48 --angRes 13 --n_blocks 1 -e 100 -lr 1e-4 -n 8 --lambda 3e-3 --batch-size 16 --test-batch-size 8 --aux-learning-rate 1e-3 --patch-size 832 832 --cuda --save --seed 1926 --gpu-id 0,1,2,3 --savepath ./checkpoint
```

Main parameters:
- `-d dataset`: Training dataset path
- `--N 48`: Number of channels
- `--angRes 13`: Angular resolution
- `--n_blocks 1`: Number of iteration blocks
- `--lambda 3e-3`: Rate-distortion parameter

### Update Entropy Model

```bash
python updata.py checkpoint_path -n checkpoint_name
```

### Model Testing

```bash
python Inference.py --dataset test_directory --output_path output_directory -p checkpoint.pth.tar
```




