import torch

from torchvision.models import vgg19
from models.parent_class import BaseModel



class VGGModel(BaseModel):
    
    """
    VGG19 model with a custom classifier.

    Attributes:
        pretrained (bool): Whether to load pretrained weights (default is True).

    Methods:
        __init__(self, pretrained: bool = True) -> None:
            Initializes the VGGModel instance.
        
        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass through the model.
        
        unfreeze_all(self) -> None:
            Unfreezes all model parameters for fine-tuning.
        
        _freeze_features(self) -> None:
            Freezes all layers except the classifier.
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        
        """
        Initializes the VGGModel instance.

        Args:
            pretrained (bool, optional): Whether to load pretrained weights (default is True).
        """
        
        super(VGGModel, self).__init__()
        
        # Load the VGG19 model
        self.base_model = vgg19(weights='IMAGENET1K_V1') if pretrained else vgg19(weights=None)
        
        # Replace classifier with custom layers
        self.base_model.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=25088, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4096),
            torch.nn.Dropout(p=0.1),

            torch.nn.Linear(in_features=4096, out_features=4096),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(4096),
            torch.nn.Dropout(p=0.1),
            
            torch.nn.Linear(in_features=4096, out_features=3)
        )
        
        if pretrained:
            self._freeze_features()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Performs the forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        return self.base_model(x)

    
    def unfreeze_all(self) -> None:
        
        """
        Unfreezes all model parameters for fine-tuning.
        """
        
        for param in self.base_model.parameters():
            param.requires_grad = True

    
    def _freeze_features(self) -> None:
        
        """
        Freezes all layers except the classifier.
        """
        
        for param in self.base_model.features.parameters():
            param.requires_grad = False
        
        for param in self.base_model.classifier.parameters():
            param.requires_grad = True