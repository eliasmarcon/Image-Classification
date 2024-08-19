#!/bin/bash

#SBATCH --job-name=conda_setup
#SBATCH -o ./logs/%x-%j.log

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
    opencv-python==4.9.0.80 \
    scikit-learn==1.5.0 \
    torchcam==0.4.0 \
    wandb==0.16.6 \
    pillow==10.3.0 \
    matplotlib==3.8.2

# Deactivate the Conda environment
conda deactivate