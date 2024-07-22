#!/bin/bash

# Initialize Conda
source /opt/conda/etc/profile.d/conda.sh
conda init bash

# Create a new conda environment
conda create -n computer_vision_project python=3.11 -y

# Activate the new environment
conda activate computer_vision_project

conda install --yes -c conda-forge pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install specific versions of packages from conda-forge
conda install --yes -c conda-forge \
    tqdm==4.66.1 \
    wandb \
    pillow==9.5.0 \
    matplotlib==3.8.2 \
    opencv \
    scikit-learn==1.3.0\
    torchcam==0.4.0

# Deactivate the Conda environment
conda deactivate