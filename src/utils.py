import torch
from pathlib import Path
from typing import Tuple


from models.vgg19_custom import CustomVGGModel
from models.vgg19 import VGGModel
from models.vgg19_adapted import VGGModelAdapted
from models.resnet import ResNet18
from models.basic_cnn import CNNBasic
from pipeline.logger import WandBLogger


def load_model(model_type: str, pretrained: bool) -> torch.nn.Module:
    
    """
    Load a model based on the specified type and pretrained option.

    Args:
    - model_type (str): The type of model to load. Options are "VGG", "VGGAdapted", "CustomVGG", "ResNet", and "CNNBasic".
    - pretrained (bool): Whether to load a pretrained model.

    Returns:
    - torch.nn.Module: The loaded model.
    
    Raises:
    - NotImplementedError: If an unsupported model_type is specified.
    """
    
    if model_type == "VGG":
        model = VGGModel(pretrained = pretrained)
        
    elif model_type == "VGGAdapted":
        model = VGGModelAdapted(pretrained = pretrained)
        
    elif model_type == "CustomVGG":
        model = CustomVGGModel(pretrained = pretrained)
        
    elif model_type == "ResNet":
        model = ResNet18(num_classes=3)
        
    elif model_type == "CNNBasic":
        model = CNNBasic(num_classes=3)
        
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented, please choose from VGG, VGGAdapted, CustomVGG.")
    
    return model


def create_logger_dir(model_type: str, pretrained: bool, data_augmentation: bool, learning_rate: float, weight_decay: float, gamma: float) -> Tuple:
    
    """
    Create a logger and directory for saving models based on the provided configuration.

    Args:
    - model_type (str): The type of model. Options are "VGG", "VGGAdapted", "CustomVGG", "ResNet", and "CNNBasic".
    - pretrained (bool): Whether the model is pretrained.
    - data_augmentation (bool): Whether data augmentation is used.
    - learning_rate (float): The learning rate.
    - weight_decay (float): The weight decay.
    - gamma (float): The gamma value.

    Returns:
    - Tuple: A tuple containing the logger and the directory for saving models.
    
    Raises:
    - NotImplementedError: If an unsupported model_type is specified.
    """
    
    # wandb logger
    wandblogger_run_name = f"{model_type}_pretrained_{str(pretrained)}_aug_{str(data_augmentation)}_lr_{learning_rate}_wd_{weight_decay}_g_{gamma}" 
    
    # create model save directory
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True)
    
    # create model save directory folder
    logger_save_dir = Path(f"saved_models/{wandblogger_run_name}")
    logger_save_dir.mkdir(exist_ok=True)
    
    if "VGGAdapted" in model_type:
        logger = WandBLogger(run_name = wandblogger_run_name, groupname="VGG19_adapted") if WandBLogger else None

    elif "CustomVGG" in model_type:
        logger = WandBLogger(run_name = wandblogger_run_name, groupname="CustomVGG") if WandBLogger else None
    
    elif "VGG" in model_type:
        logger = WandBLogger(run_name = wandblogger_run_name, groupname="VGG19_no_change") if WandBLogger else None
    
    elif "ResNet" in model_type:
        logger = WandBLogger(run_name = wandblogger_run_name, groupname="ResNet") if WandBLogger else None
        
    elif "CNNBasic" in model_type:
        logger = WandBLogger(run_name = wandblogger_run_name, groupname="CNNBasic") if WandBLogger else None
        
    else:
        raise NotImplementedError(f"Model type {model_type} is not implemented, so logger cannot be created.")
    
    return logger, logger_save_dir
