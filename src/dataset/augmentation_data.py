import glob
import os
import random
import shutil
import torch
import matplotlib.pyplot as plt

from torchvision import transforms, io
from PIL import Image


# Set random seed for reproducibility
torch.manual_seed(100)
torch.cuda.manual_seed(100)
random.seed(100)
FOLDER_TYPES = ["Axe", "Book", "Hammer"] # 0 = Axe, 1 = Book, 2 = Hammer


def read_augmented_data(base_image_path: str = "./dataset/") -> tuple[list[str], list[str]]:
    
    """
    Reads augmented image paths and their corresponding labels.

    Args:
    - base_image_path (str): Base path to the dataset directory.

    Returns:
    - tuple: A tuple containing a list of augmented image paths and a list of corresponding labels.
    """
    
    aug_image_paths = []
    aug_labels = []
    
    for folder in ["Axe", "Hammer"]:
        
        temp_paths = [s.replace("\\", "/") for s in glob.glob(base_image_path + folder + "_augmented/*.jpg")]
        # sort images by number
        temp_paths = sorted(temp_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
        aug_image_paths.extend(temp_paths)
        aug_labels += [folder] * len(temp_paths)
        
    return aug_image_paths, aug_labels


def read_data(base_image_path: str = "./dataset/", size: float = 1, augment: bool = False) -> tuple[list[str], list[str]]:
    
    """
    Reads image paths and their corresponding labels, optionally including augmented data.

    Args:
    - base_image_path (str): Base path to the dataset directory.
    - size (float): Proportion of 'Book' images to include.
    - augment (bool): Whether to include augmented data.

    Returns:
    - tuple: A tuple containing a list of image paths and a list of corresponding labels.
    """
    
    all_image_paths = []
    labels = []
    
    for folder in FOLDER_TYPES:
                
        temp_paths = [s.replace("\\", "/") for s in glob.glob(base_image_path + folder + "/*.jpg")]
        # sort images by number
        temp_paths = sorted(temp_paths, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
        
        # reduce the number of book images
        if folder == "Book":
            temp_paths = temp_paths[: int(len(temp_paths) * size)]
        
        all_image_paths.extend(temp_paths)
        labels += [folder] * len(temp_paths)
        
    if augment:
        
        aug_paths, aug_labels = read_augmented_data(base_image_path)
        all_image_paths.extend(aug_paths)
        labels.extend(aug_labels)

    return all_image_paths, labels


def create_testing_batch(number_of_images: int = 30) -> None:
    
    """
    Creates a testing dataset by randomly selecting images from each class.

    Args:
    - number_of_images (int): Number of images to select from each class.
    """
    
    if os.path.exists("./dataset_testing/"):
        print("The dataset_testing folder already exists and therefore no additional images will be added")
        
    else:
        
        image_paths, _ = read_data(size=1)
        
        # create the folder
        os.mkdir("./dataset_testing/")
        
        # create the subfolders
        for folder in FOLDER_TYPES:
            os.mkdir(f"./dataset_testing/{folder}")
            
        # take X images from each folder randomly and move them to the dataset_testing folder
        for folder in FOLDER_TYPES:
            
            temp_images = [image_path for image_path in image_paths if folder in image_path]
            random.shuffle(temp_images)
                        
            for i in range(number_of_images):
                
                shutil.move(temp_images.pop(0), f"./dataset_testing/{folder}/{i}.jpg")
                

def resize_or_pad_image(image_path, output_size=(224, 224)):
    
    """
    Resize images larger than the output size to fit within the output size while maintaining aspect ratio,
    and pad images smaller than the output size to center them.
    
    Args:
    - image_path (str): Path to the input image.
    - output_size (tuple): Desired output size as (width, height). Default is (224, 224).
    
    Returns:
    - transformed_image (Tensor): Transformed image as a PyTorch tensor.
    """
    
    # Load image
    image = Image.open(image_path)
    
    # Get original image dimensions
    width, height = image.size

    # Calculate aspect ratio
    aspect_ratio = width / height

    # Determine target size respecting aspect ratio
    if width > height:
        new_width = output_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = output_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Calculate padding to center the image
    pad_width = output_size[0] - new_width
    pad_height = output_size[1] - new_height
    padding = (pad_width // 2, pad_height // 2, pad_width - (pad_width // 2), pad_height - (pad_height // 2))

    # Apply padding to ensure the output size and center the image
    padded_resized_image = transforms.functional.pad(resized_image, padding, fill=0, padding_mode='constant')

    # Convert PIL Image to PyTorch tensor
    image_tensor = transforms.ToTensor()(padded_resized_image)
    
    if image_tensor.shape[0] < 3:
        # If the image is grayscale, convert it to RGB
        image_tensor = torch.cat([image_tensor] * (3 // image_tensor.shape[0]), dim=0)
    
    return image_tensor


def apply_random_augmentation(image_tensor: torch.Tensor) -> torch.Tensor:
    
    """
    Apply random augmentation to an image tensor using PyTorch transforms.

    Args:
    - image_tensor (torch.Tensor): Input image tensor.

    Returns:
    - torch.Tensor: Augmented image tensor.
    """
    
    # Define the augmentation choices
    augmentation_choices = [
        transforms.RandomRotation(degrees=40),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomVerticalFlip(p=0.8),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.GaussianBlur(kernel_size=3)
    ]
    
    # Randomly choose a subset of augmentations
    num_augmentations = random.choice([1, 2, 3, 4, 5, 6])
    augmentation_choice = random.sample(augmentation_choices, num_augmentations)
    
    # Apply the augmentations
    for augmentation in augmentation_choice:
        image_tensor = augmentation(image_tensor)
    
    return image_tensor


def create_augmented_data(image_paths: list[str], labels: list[str]) -> None:
    
    """
    Create augmented data for undersampled classes by applying random augmentations.

    Args:
    - image_paths (list): List of image paths.
    - labels (list): List of corresponding labels.
    """
    
    # Load images and apply augmentation
    images = [io.read_image(image_path) for image_path in image_paths]

    # Get the count of images in each class
    overall_count = {label: labels.count(label) for label in set(labels)}
    
    # Get the classes with the fewest images
    undersampled_classes = [label for label, count in overall_count.items() if count < max(overall_count.values())]
    
    # Augment the undersampled class until the number of images in the class is equal to the number of images in the majority class
    for undersampled_class in undersampled_classes:
        # Get indices of images in the undersampled class
        indices = [ind for ind, label in enumerate(labels) if label == undersampled_class]
        number_of_augmented_images = max(overall_count.values()) - overall_count[undersampled_class]

        # get the biggest image name for the undersampled class
        biggest_image_number = max([int(os.path.splitext(os.path.basename(path))[0]) for path in image_paths if undersampled_class in path])
 
        for i in range(number_of_augmented_images):
            # Randomly select an image from the undersampled class
            image_tensor = images[random.choice(indices)]
            
            # Apply random augmentation
            augmented_image = apply_random_augmentation(image_tensor)
            
            # Ensure image tensor is in RGB format
            if augmented_image.shape[0] < 3:
                # If the image is grayscale, convert it to RGB
                augmented_image = torch.cat([augmented_image] * (3 // augmented_image.shape[0]), dim=0)
            
            # Convert tensor to numpy array and save as image
            augmented_image_np = augmented_image.permute(1, 2, 0).numpy()
            image_number = biggest_image_number + i
            prefix = '000' if image_number < 10 else ('00' if image_number < 100 else ('0' if image_number < 1000 else ''))
            filename = f'{prefix}{image_number}.jpg'
            
            save_directory = f'./dataset/{undersampled_class}_augmented'
            
            if not os.path.isdir(save_directory):
                os.makedirs(save_directory)
            
            plt.imsave(os.path.join(save_directory, filename), augmented_image_np)
            
            
def plot_augmented_images(number_of_images: int = 3) -> None:
    
    """
    Plot and save a specified number of augmented images for each class.

    Args:
    - number_of_images (int): Number of images to plot for each class.
    """
    
    image_paths, _ = read_augmented_data()
    
    # select number of images for each class
    axe_paths = [path for path in image_paths if "Axe" in path]
    random.shuffle(axe_paths)
    axe_paths = axe_paths[:number_of_images]
    
    hammer_paths = [path for path in image_paths if "Hammer" in path]
    random.shuffle(hammer_paths)
    hammer_paths = hammer_paths[:number_of_images]
    
    fig, ax = plt.subplots(2, number_of_images, figsize=(20, 10))
    
    for i in range(number_of_images):
            
        axe_image = Image.open(axe_paths[i])
        hammer_image = Image.open(hammer_paths[i])
        
        ax[0, i].imshow(axe_image)
        ax[1, i].imshow(hammer_image)
        
        ax[0, i].axis("off")
        ax[1, i].axis("off")
        
        ax[0, i].set_title("Axe")
        ax[1, i].set_title("Hammer")
        
    plt.tight_layout()
    plt.savefig("./readme_images/dataset_images/augmented_images.png")
    

    