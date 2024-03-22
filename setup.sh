#!/bin/bash

# GCC
sudo apt update
sudo apt -y install build-essential

# Conda
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda remove -n caries_staging -y --all
conda create -n caries_staging -y python=3.11
conda activate caries_staging

# PyTorch
pip3 install torch torchvision torchaudio
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-2.2.1+cu121.html

# MMEngine, MMCV
pip3 install -U openmim
mim install mmengine
mim install "mmcv==2.1.0"

# MMDetection
pip3 install -v -e $(dirname "$0")/mmdetection

# Pip requirements
pip3 install -r $(dirname "$0")/requirements.txt
