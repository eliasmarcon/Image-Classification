import torch

from models.parent_class import BaseModel



class CNNBasic(BaseModel):
    
    """
    CNNBasic model architecture.

    Args:
        num_classes (int): Number of output classes.

    Attributes:
        conv1_1 (torch.nn.Conv2d): First convolutional layer.
        conv1_1_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv1_1.
        conv1_2 (torch.nn.Conv2d): Second convolutional layer in the first block.
        conv1_2_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv1_2.
        pool1 (torch.nn.MaxPool2d): Max pooling layer after the first block.
        conv2_1 (torch.nn.Conv2d): First convolutional layer in the second block.
        conv2_1_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv2_1.
        conv2_2 (torch.nn.Conv2d): Second convolutional layer in the second block.
        conv2_2_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv2_2.
        pool2 (torch.nn.MaxPool2d): Max pooling layer after the second block.
        conv3_1 (torch.nn.Conv2d): First convolutional layer in the third block.
        conv3_1_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv3_1.
        conv3_2 (torch.nn.Conv2d): Second convolutional layer in the third block.
        conv3_2_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv3_2.
        pool3 (torch.nn.MaxPool2d): Max pooling layer after the third block.
        conv4_1 (torch.nn.Conv2d): First convolutional layer in the fourth block.
        conv4_1_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv4_1.
        conv4_2 (torch.nn.Conv2d): Second convolutional layer in the fourth block.
        conv4_2_bn (torch.nn.BatchNorm2d): Batch normalization layer for conv4_2.
        pool4 (torch.nn.MaxPool2d): Max pooling layer after the fourth block.
        adaptive_max_pool (torch.nn.AdaptiveMaxPool2d): Adaptive max pooling layer.
        dense_1 (torch.nn.Linear): First fully connected layer.
        dense_2 (torch.nn.Linear): Second fully connected layer.
        gelu (torch.nn.GELU): GELU activation function.
        dropout (torch.nn.Dropout): Dropout layer for regularization.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the network.

    """
    
    def __init__(self, num_classes: int = 3) -> None:
        
        """
        Initializes CNNBasic with specified number of output classes.

        Args:
            num_classes (int): Number of output classes.
        """
        
        super(CNNBasic, self).__init__()
            
        # Define the convolutional layers with batch normalization and ReLU activation
        # Convolution 1
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_1_bn = torch.nn.BatchNorm2d(64)
        self.conv1_2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2_bn = torch.nn.BatchNorm2d(64)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_1_bn = torch.nn.BatchNorm2d(128)
        self.conv2_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2_bn = torch.nn.BatchNorm2d(128)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Convolution 3
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_1_bn = torch.nn.BatchNorm2d(256)
        self.conv3_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2_bn = torch.nn.BatchNorm2d(256)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)

        # Convolution 4 (newly added)
        self.conv4_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_1_bn = torch.nn.BatchNorm2d(512)
        self.conv4_2 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2_bn = torch.nn.BatchNorm2d(512)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)

        # Adaptive Max Pooling
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d((1, 1))

        # Fully connected layers
        self.dense_1 = torch.nn.Linear(512, 128)
        self.dense_2 = torch.nn.Linear(128, num_classes)
        
        self.gelu = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.5)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass through the CNNBasic model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        
        # Forward pass through the first convolutional block
        x = self.conv1_1(x)
        x = self.conv1_1_bn(x)
        x = self.gelu(x)
        
        x = self.conv1_2(x)
        x = self.conv1_2_bn(x)
        x = self.gelu(x)
        
        x = self.pool1(x)
        
        # Forward pass through the second convolutional block
        x = self.conv2_1(x)
        x = self.conv2_1_bn(x)
        x = self.gelu(x)
        
        x = self.conv2_2(x)
        x = self.conv2_2_bn(x)
        x = self.gelu(x)
        
        x = self.pool2(x)
        
        # Forward pass through the third convolutional block
        x = self.conv3_1(x)
        x = self.conv3_1_bn(x)
        x = self.gelu(x)
        
        x = self.conv3_2(x)
        x = self.conv3_2_bn(x)
        x = self.gelu(x)
        
        x = self.pool3(x)
        
        # Forward pass through the fourth convolutional block
        x = self.conv4_1(x)
        x = self.conv4_1_bn(x)
        x = self.gelu(x)
        
        x = self.conv4_2(x)
        x = self.conv4_2_bn(x)
        x = self.gelu(x)
        
        x = self.pool4(x)
        
        # Adaptive Max Pooling
        x = self.adaptive_max_pool(x)
        
        # Flatten the feature maps
        x = torch.flatten(x, 1)
        
        # Forward pass through the fully connected layers
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dropout(x)  # Apply dropout for regularization
        
        x = self.dense_2(x)
        
        return x
