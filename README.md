# text3D2video

Controllable 4D-guided video generation with T2I diffusion models using 3D consistent diffusion features.

## Repository Structure

- `data` contains some example meshes, animations, etc
- `notebooks` contains some interactive notebooks
- `scripts` contains the main entrypoints to our program
- `text3d2video` contains reusable functions defined in our library

## Getting Started

First create and activate a conda environment

```bash
conda create -n thesis python=3.9
conda activate thesis
```

Install Pytorch 1.12

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# for latest pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Install Pytorch3D

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

Install `faiss` 

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

And install pip packages from `requirements.txt`

```bash
pip install -r requirements.txt
```