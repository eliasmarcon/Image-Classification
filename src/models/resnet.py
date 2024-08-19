import torch

from models.parent_class import BaseModel



class BasicBlock(torch.nn.Module):
    
    """
    Basic building block for ResNet.

    Attributes:
        expansion (int): Expansion factor for the block.
            By default, it is set to 1.

    Methods:
        __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
            Initializes the BasicBlock.

        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass through the BasicBlock.
    """
    
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        
        """
        Initializes the BasicBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride size for the convolutional layers (default is 1).
        """
        
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # Shortcut connection to downsample residual
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Performs the forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet18(BaseModel):
    
    """
    ResNet-18 architecture based on torchvision's implementation.

    Attributes:
        in_channels (int): Number of input channels for the first layer.

    Methods:
        __init__(self, num_classes: int = 3) -> None:
            Initializes the ResNet18 model.

        forward(self, x: torch.Tensor) -> torch.Tensor:
            Performs the forward pass through the ResNet18 model.

        _make_layer(self, block: torch.nn.Module, out_channels: int, blocks: int, stride: int) -> torch.nn.Sequential:
            Constructs each layer of the ResNet18 model.

        _initialize_weights(self) -> None:
            Initializes weights of the model layers.
    """

    def __init__(self, num_classes: int = 3) -> None:
        
        """
        Initializes the ResNet18 model.

        Args:
            num_classes (int, optional): Number of output classes (default is 3).
        """
        
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512 * BasicBlock.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Performs the forward pass through the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
    def _make_layer(self, block: torch.nn.Module, out_channels: int, blocks: int, stride: int) -> torch.nn.Sequential:
        
        """
        Constructs each layer of the ResNet18 model.

        Args:
            block (torch.nn.Module): Type of block to be used (BasicBlock in this case).
            out_channels (int): Number of output channels.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride size for the convolutional layers.

        Returns:
            torch.nn.Sequential: Sequential layer of blocks.
        """
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
            self.in_channels = out_channels * block.expansion

        return torch.nn.Sequential(*layers)


    def _initialize_weights(self) -> None:
        
        """
        Initializes weights of the model layers.
        """
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
                