#!/bin/bash

#SBATCH --job-name=cv
#SBATCH -o ./logs/%x-%j.log
#SBATCH --exclusive

# Assign the hyperparameter file argument
file_number="$1"
PARAM_FILE="./parameters/hyperparameters_${file_number}.txt"


# Check if the parameter file exists
if [ ! -f "$PARAM_FILE" ]; then
  echo "Parameter file 'parameters/$PARAM_FILE' not found!"
  exit 1
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate computer_vision_project

# Read hyperparameters from the file into an array
IFS=$'\n' read -d '' -r -a hyperparameters < "$PARAM_FILE"

# Loop over each set of hyperparameters
for params in "${hyperparameters[@]}"; do
    set -- $params
    model_type=$1
    pretrained=$2
    num_epochs=$3
    batch_size=$4
    learning_rate=$5
    weight_decay=$6
    gamma=$7
    data_augmentation=$8
    plotting=$9

    # Remove newline character using parameter expansion
    plotting="${plotting%"${plotting##*[![:space:]]}"}"


    # Convert flags to correct format
    if [ "$pretrained" == "True" ]; then
      PRETRAINED="--pretrained"
    else
      PRETRAINED=""
    fi

    if [ "$data_augmentation" == "True" ]; then
      DATA_AUGMENTATION="--data_augmentation"
    else
      DATA_AUGMENTATION=""
    fi

    if [ "$plotting" == "True" ]; then
      PLOTTING="--plots"
    else
      PLOTTING=""
    fi

    # Run the training script
    srun python ./src/main.py \
                              --model_type "$model_type" \
                              $PRETRAINED \
                              --num_epochs "$num_epochs" \
                              --batch_size "$batch_size" \
                              --learning_rate "$learning_rate" \
                              --weight_decay "$weight_decay" \
                              --gamma "$gamma" \
                              $DATA_AUGMENTATION \
                              $PLOTTING
done

# Deactivate Conda environment
conda deactivate

# Delete Folder wandb if needed
# rm -rf wandb
