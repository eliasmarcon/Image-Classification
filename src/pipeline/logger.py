import torch
import wandb

from pipeline import config


class WandBLogger:
    
    """
    A class for logging experiments using Weights & Biases (WandB).

    Attributes:
        enabled (bool): Flag to enable or disable logging.
        run (wandb.wandb_run.Run): The WandB run instance.
    """

    def __init__(self, enabled=True, 
                 model: torch.nn.Module = None, 
                 run_name: str = None,
                 groupname: str = None) -> None:
        
        """
        Initializes the WandBLogger instance.

        Args:
            enabled (bool): Whether to enable WandB logging.
            model (torch.nn.Module): The model to watch.
            run_name (str): Custom name for the WandB run.
        """
        
        self.enabled = enabled
        self.run = None
        self.run_name = run_name

        if self.enabled:
            
            wandb.login(key=config.WANDB_KEY, relogin=True)
            
            self.run = wandb.init(entity=config.WANDB_ENTITY,
                                  project=config.WANDB_PROJECT,
                                  group=groupname) 

            if run_name is None:
                self.run.name = self.run.id    
            else:
                self.run.name = run_name 

            if model is not None:
                self.watch(model)


    def watch(self, model: torch.nn.Module, log_freq: int = 1) -> None:
        
        """
        Watches the model to log gradients and parameters.

        Args:
            model (torch.nn.Module): The model to watch.
            log_freq (int): Frequency of logging gradients and parameters.
        """
        
        if self.enabled:
            wandb.watch(model, log="all", log_freq=log_freq)


    def log(self, log_dict: dict, commit: bool = True, step: int = None) -> None:
        
        """
        Logs metrics to WandB.

        Args:
            log_dict (dict): Dictionary of metrics to log.
            commit (bool): Whether to commit the log entry.
            step (int): Step number for logging.
        """
        
        if self.enabled:
            
            if step is not None:
                wandb.log(log_dict, commit=commit, step=step)
        
            else:
                wandb.log(log_dict, commit=commit)


    def finish(self) -> None:
        
        """
        Finishes the WandB run.
        """
        
        if self.enabled:
            wandb.finish()
