#!/bin/bash

#SBATCH --job-name=cv
#SBATCH -o ./logs/%x-%j.log
#SBATCH --exclusive

source /opt/conda/etc/profile.d/conda.sh
conda activate computer_vision_project

# Define variables for positional arguments
MODEL_TYPE="$1"

if [ "$2" == "True" ]; then
  PRETRAINED="--pretrained"
else
  PRETRAINED=""
fi

NUM_EPOCHS="$3"
BATCH_SIZE="$4"
LEARNING_RATE="$5"
WEIGHT_DECAY="$6"
GAMMA="$7"

if [ "$8" == "True" ]; then
  DATA_AUGMENTATION="--data_augmentation"
else
  DATA_AUGMENTATION=""
fi

if [ "$9" == "True" ]; then
  PLOTTING="--plots"
else
  PLOTTING=""
fi

# Run the training script
srun python ./src/main.py \
                            --model_type "$MODEL_TYPE" \
                            $PRETRAINED \
                            --num_epochs "$NUM_EPOCHS" \
                            --batch_size "$BATCH_SIZE" \
                            --learning_rate "$LEARNING_RATE" \
                            --weight_decay "$WEIGHT_DECAY" \
                            --gamma "$GAMMA" \
                            $DATA_AUGMENTATION \
                            $PLOTTING

# Deactivate Conda environment
conda deactivate

# Delete Folder wandb
rm -rf wandb