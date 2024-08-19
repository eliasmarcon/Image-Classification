import torch

from torchvision.models import vgg19
from models.parent_class import BaseModel


            
class CustomVGGModel(BaseModel):
    
    """
    Custom VGG19 model with modified feature extraction and custom classifier layers.

    Attributes:
        pretrained (bool): Whether to load pretrained weights (default is True).

    Methods:
        __init__(self, pretrained: bool = True) -> None:
            Initializes the CustomVGGModel instance.
        
        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass through the model.
        
        unfreeze_all(self) -> None:
            Unfreezes all model parameters for fine-tuning.
        
        _initialize_weights(self) -> None:
            Initializes the weights of the custom layers.
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        
        """
        Initializes the CustomVGGModel instance.

        Args:
            pretrained (bool, optional): Whether to load pretrained weights (default is True).
        """
        
        super(CustomVGGModel, self).__init__()
        # Load the base VGG19 model
        self.base_model = vgg19(weights='IMAGENET1K_V1') if pretrained else vgg19(weights=None)
                
        # Split the base model to use features up to block4 conv4
        self.base_features = torch.nn.Sequential(
            *list(self.base_model.features[:27])
        )
        
        # Custom convolutional layers
        self.custom_conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(512, 128, kernel_size=1, padding='same'),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            torch.nn.GELU(),
            torch.nn.Conv2d(128, 512, kernel_size=1, padding='same'),
            torch.nn.GELU(),
            torch.nn.Conv2d(512, 1024, kernel_size=1, padding='valid'),
            torch.nn.GELU(),
            torch.nn.Conv2d(1024, 1024, kernel_size=3, padding='same'),
            torch.nn.GELU(),
        )
        
        # Pooling layer after the custom convolutional layers
        self.pooling_layer = torch.nn.AdaptiveMaxPool2d((1, 1))
        
        # Custom classification layers
        self.custom_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 1024),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 512),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, 256),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Linear(256, 3)
        )
        
        # Freeze layers if pretrained
        if pretrained:    
            for param in self.base_features.parameters():
                param.requires_grad = False


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Performs the forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.base_features(x)
        x = self.custom_conv_layers(x)
        x = self.pooling_layer(x)
        x = self.custom_layers(x)
        
        return x
    
    
    def unfreeze_all(self) -> None:
        
        """
        Unfreezes all model parameters for fine-tuning.
        """
        
        for param in self.base_model.parameters():
            param.requires_grad = True