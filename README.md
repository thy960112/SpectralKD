# SpectralKD: A Unified Framework for Interpreting and Distilling Vision Transformers via Spectral Analysis

## Introduction

We propose SpectralKD, a spectral analysis framework that unifies interpretation and distillation of Vision Transformers, achieving state-of-the-art performance without trainable parameters while revealing fundamental distillation dynamics.

## Run

### Environment

```
pytorch==2.4.0
timm==1.0.11
```

### Data Preparation

Download and extract ImageNet train and val images from <http://image-net.org/>.  
The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training on ImageNet-1K

**Note:** Pretrained teacher `cait_s24_224` and `swin_small_patch4_window7_224.ms_in1k` will be download automatically from timm.

#### DeiT-Tiny

To train a DeiT-Tiny student with a Cait-S24 teacher, run:

```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py --output_dir Output/DeiT-Tiny --data-path ILSVRC2012 --model deit_tiny_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 0.9 --w-fft 0.2 --s-id 0 1 6 7 8 9 10 11 --t-id 0 1 18 19 20 21 22 23 --drop-path 0 --batch-size 256 --num_workers 16 --epochs 400
```

#### DeiT-Small

To train a DeiT-Small student with a Cait-S24 teacher, run:

```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py --output_dir Output/DeiT-Small --data-path ILSVRC2012 --model deit_small_patch16_224 --teacher-model cait_s24_224 --distillation-type soft --distillation-alpha 0.9 --w-fft 0.2 --s-id 0 1 6 7 8 9 10 11 --t-id 0 1 18 19 20 21 22 23 --drop-path 0 --batch-size 256 --num_workers 16 --epochs 500 
```

#### Swin-Tiny

To train a Swin-Tiny student with a Swin-Small teacher, run:

```Shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 main.py --output_dir Output/Swin-Tiny --data-path ILSVRC2012 --model swin_tiny_patch4_window7_224 --teacher-model swin_small_patch4_window7_224.ms_in1k --distillation-type soft --distillation-alpha 0.9 --w-fft 0.05 --s-id 0 1 2 3 --t-id 0 1 2 3 --drop-path 0 --batch-size 256 --num_workers 16 --epochs 500 --warmup-epochs 20
```

## Acknowledgment

This repo is based on [manifold-distillation](https://github.com/Hao840/manifold-distillation), [DeiT](https://github.com/facebookresearch/deit) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models).


