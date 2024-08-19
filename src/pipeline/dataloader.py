import torch

from torchvision import transforms
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Own Modules
import dataset.augmentation_data as data_source


# Set random seed for reproducibility
torch.manual_seed(100)
torch.cuda.manual_seed(100)



class DataLoader(Dataset):
    
    def __init__(self, batch_size, image_paths, labels) -> None:
        
        """
        Initialize the DataLoader with batch size, image paths, and labels.

        Args:
        - batch_size (int): The size of batches.
        - image_paths (list[str]): List of paths to the images.
        - labels (list[str]): List of labels corresponding to the images.
        """
        
        self.batch_size = batch_size
        self.image_paths = image_paths
        self.labels = labels
        
        
    def get_dataset(self, dataset_type = "train") -> torch.utils.data.DataLoader:
        
        """
        Get the DataLoader for the specified dataset type.

        Args:
        - dataset_type (str): The type of dataset ("train", "test"). Defaults to "train".

        Returns:
        - DataLoader or tuple: DataLoader for the test set, or tuple of DataLoaders for train and validation sets.
        """
        
        # resize or pad images
        images = self._resize_pad()
        transformed_images = [self._transform(image) for image in images]
        
        if dataset_type == "train":
            
            train_images, val_images, train_labels, val_labels = self._split_data(transformed_images)

            train_set = self._apply_mapping(train_images, train_labels)
            val_set = self._apply_mapping(val_images, val_labels)
            
            train_loader = self._get_train_loader(train_set)
            val_loader = self._get_val_loader(val_set)
            
            return train_loader, val_loader
        
        elif dataset_type == "test":
            
            test_set = self._apply_mapping(transformed_images, self.labels)
            test_loader = self._get_test_loader(test_set)
            
            return test_loader
        
        else:
            
            raise ValueError(f"Invalid dataset type {dataset_type}")        
           
                
    def _resize_pad(self) -> list[torch.Tensor]:
        
        """
        Resize or pad images.

        Returns:
        - list[torch.Tensor]: List of resized or padded images as tensors.
        """
        
        return [data_source.resize_or_pad_image(path) for path in self.image_paths]
            
            
    def _transform(self, image) -> torch.Tensor:
        
        """
        Apply transformations to the image.

        Args:
        - image (torch.Tensor): Image tensor to be transformed.

        Returns:
        - torch.Tensor: Transformed image tensor.
        """
        
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])
        
        return transform(image)
    
    
    def _split_data(self, transformed_images) -> tuple:
        
        """
        Split data into training and validation sets.

        Args:
        - transformed_images (list[torch.Tensor]): List of transformed image tensors.

        Returns:
        - tuple: Split data (train_images, val_images, train_labels, val_labels).
        """
        
        # Split data into training and validation sets
        return train_test_split(transformed_images, self.labels, test_size=0.2, random_state=42)
    
    
    def _apply_mapping(self, transformed_images, labels) -> TensorDataset:
        
        """
        Apply label mapping and create a TensorDataset.

        Args:
        - transformed_images (list[torch.Tensor]): List of transformed image tensors.
        - labels (list[str]): List of labels.

        Returns:
        - TensorDataset: TensorDataset containing the transformed images and mapped labels.
        """
        
        # Define label mapping
        mapping = {'Axe' : 0, 'Book' : 1, 'Hammer': 2}
                
        # Apply mapping
        labels = [mapping[label] for label in labels]    
        # Convert labels to tensor
        labels = torch.tensor(labels)
        # Convert images to tensor
        transformed_images = torch.stack(transformed_images, dim=0)
        
        return TensorDataset(transformed_images, labels)
    
    
    def _get_train_loader(self, train_data) -> torch.utils.data.DataLoader:
        
        """
        Get the DataLoader for training data.

        Args:
        - train_data (TensorDataset): TensorDataset for training data.

        Returns:
        - DataLoader: DataLoader for training data.
        """
        
        return torch.utils.data.DataLoader(
                                            train_data,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            num_workers=6,
                                            drop_last=False,
                                            persistent_workers=True
                                        )
    
    
    def _get_val_loader(self, val_data) -> torch.utils.data.DataLoader:
        
        """
        Get the DataLoader for validation data.

        Args:
        - val_data (TensorDataset): TensorDataset for validation data.

        Returns:
        - DataLoader: DataLoader for validation data.
        """
        
        return torch.utils.data.DataLoader(
                                            val_data,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=6,
                                            drop_last=False,
                                            persistent_workers=True
                                        )
      
        
    def _get_test_loader(self, test_data) -> torch.utils.data.DataLoader:
        
        """
        Get the DataLoader for test data.

        Args:
        - test_data (TensorDataset): TensorDataset for test data.

        Returns:
        - DataLoader: DataLoader for test data.
        """
        
        return torch.utils.data.DataLoader(
                                            test_data,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            num_workers=0,
                                            drop_last=False
                                        )