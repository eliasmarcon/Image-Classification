import torch

from pathlib import Path
from abc import ABC

class BaseModel(torch.nn.Module, ABC):
    
    """
    Base class for PyTorch models.

    This class inherits from torch.nn.Module and abc.ABC (Abstract Base Class).
    It provides methods for saving and loading model checkpoints.

    Attributes:
        None

    Methods:
        save(save_dir: Path, suffix: str = None) -> None:
            Saves the model state dictionary to a file.

        load(path: Path) -> None:
            Loads the model state dictionary from a file.
    """

    def save(self, save_dir: Path, suffix: str = None) -> None:
        
        """
        Saves the model state dictionary to a file.

        Args:
            save_dir (Path): Directory where the model checkpoint will be saved.
            suffix (str, optional): Suffix to append to the filename. Defaults to None.
        """
        
        filename = save_dir / (f"{suffix}.pt" if suffix else "model.pt")
        torch.save(self.state_dict(), filename)


    def load(self, path: Path) -> None:
        
        """
        Loads the model state dictionary from a file.

        Args:
            path (Path): Path to the saved model checkpoint.
        """
        
        self.load_state_dict(torch.load(path, weights_only=True))
