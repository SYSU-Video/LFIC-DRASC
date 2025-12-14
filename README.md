# LFIC-DRASC - Light Field Image Compression Neural Network

![GitHub stars](https://img.shields.io/github/stars/SYSU-Video/LFIC-DRASC?style=social)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
[![Paper](https://img.shields.io/badge/Paper-arxiv'24-b31b1b.svg)](https://arxiv.org/abs/2409.11711)
![License](https://img.shields.io/github/license/SYSU-Video/LFIC-DRASC)
![Last commit](https://img.shields.io/github/last-commit/SYSU-Video/LFIC-DRASC)

LFIC-DRASC: Deep Light Field Image Compression Using Disentangled Representation and Asymmetrical Strip Convolution \
[[paper]](https://ieeexplore.ieee.org/document/11068206) [[code]](https://github.com/SYSU-Video/LFIC-DRASC) \
[Shiyu Feng](https://orcid.org/0009-0002-9085-727X), [Yun Zhang](https://codec.siat.ac.cn/yunzhang/), [Linwei Zhu](https://zhulinweicityu.github.io/), [Sam Kwong](https://scholars.ln.edu.hk/en/persons/sam-tak-wu-kwong) \
*IEEE Transactions on Broadcasting (TBC), 2025*

## Project Introduction

LFIC-DRASC is a deep learning model for light field image compression, which maintains spatial consistency through a special network structure, improving compression efficiency and visual quality.

## Abstract
Light-Field (LF) image is emerging 4D data of light rays that is capable of realistically presenting spatial and angular information of 3D scene. However, the large data volume of LF images becomes the most challenging issue in real-time processing, transmission, and storage. In this paper, we propose an end-to-end deep LF Image Compression method Using Disentangled Representation and Asymmetrical Strip Convolution (LFIC-DRASC) to improve coding efficiency. Firstly, we formulate the LF image compression problem as learning a disentangled LF representation network and an image encodingdecoding network. Secondly, we propose two novel feature extractors that leverage the structural prior of LF data by integrating features across different dimensions. Meanwhile, disentangled LF representation network is proposed to enhance the LF feature disentangling and decoupling. Thirdly, we propose the LFIC-DRASC for LF image compression, where two Asymmetrical Strip Convolution (ASC) operators, i.e., horizontal and vertical, are proposed to capture long-range correlation in LF feature space. These two ASC operators can be combined with the square convolution to further decouple LF features, which enhances the model’s ability in representing intricate spatial relationships. Experimental results demonstrate that the proposed LFIC-DRASC achieves an average of 20.5% bit rate reductions compared with the state-of-the-art methods. Source code and pre-trained models of LFIC-DRASC are available at https://github.com/SYSU-Video/LFIC-DRASC.


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

Note: We have retrained the models on RTX4090 and updated their checkpoints, which are provided in the ckpt folder. 



