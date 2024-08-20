# Thesis

Controllable 4D-guided video generation with image diffusion models, with 3D consistent diffusion features.

## Getting Started

First create and activate a conda environment

```bash
conda create -n thesis python=3.9
conda activate thesis
```

Install Pytorch 1.12

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Install Pytorch3D

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

And install pip packages from `requirements.txt`

```bash
pip install -r requirements.txt
```

