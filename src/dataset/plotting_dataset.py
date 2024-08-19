import os
import matplotlib.pyplot as plt
import cv2

from torchvision import transforms
from augmentation_data import resize_or_pad_image


FOLDER_TYPES = ['Axe', 'Book', 'Hammer']

def plot_dataset_size(save_path: str = None) -> None:
    
    """
    Plot the number of images in each dataset folder and save the plot.

    Args:
    - save_path (str): Path to save the plot. If the file already exists, it will not be overwritten.
    """
    
    if os.path.exists(save_path):
        print("The file already exists")
        return
    
    # show the amount of images in the dataset
    folder_lengths = []

    for folder in FOLDER_TYPES:
        # save the amount of images in the folder
        folder_length = len(os.listdir(f"../dataset/{folder}"))
        folder_lengths.append(folder_length)      

    # plot the amount of images in the dataset
    plt.figure(figsize=(10, 8))
    plt.bar(FOLDER_TYPES, folder_lengths)

    # show values on top of the bars
    for i in range(len(FOLDER_TYPES)):
        plt.text(i, folder_lengths[i], folder_lengths[i], ha='center', va='bottom')

    plt.title('Amount of images in the dataset')
    plt.ylabel('Amount of images')
    
    # Save the plot
    plt.savefig(save_path)

    plt.show()


def reduce_dataset() -> None:
    
    """
    Reduce the number of images in each dataset folder to a maximum of 1500 images.
    """
    
    for folder in FOLDER_TYPES:
    
        delete_counter = 0
        
        if len(os.listdir(f"./dataset/{folder}")) > 1500:
            
            # get all images in the folder over 1500
            images = os.listdir(f"./dataset/{folder}")[1500:]
            
            for image in images:
                
                # delete the image
                os.remove(f"./dataset/{folder}/{image}")
                delete_counter += 1
            
            print(f"Deleted {delete_counter} images from the {folder} folder")
            
            
def plot_image_format_statistics() -> None:
    
    """
    Plot statistics of image formats (dimensions and channels) for each dataset folder and save the plots.
    """
    
    for folder in FOLDER_TYPES:
    
        folder_path = os.path.join('./dataset/', folder)
        
        # Initialize counters for image formats
        format_counts = {}

        # Iterate over files in the folder
        for filename in os.listdir(folder_path):
            
            # Get the file extension
            _, ext = os.path.splitext(filename)
            
            # Skip if the file is not an image
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            
            # Read the image
            img = cv2.imread(os.path.join(folder_path, filename))
            
            # Get image format (number of channels)
            format_info = f"{img.shape[0]}x{img.shape[1]}x{img.shape[2]}"
            
            # Increment the counter for the format
            if format_info in format_counts:
                format_counts[format_info] += 1
            else:
                format_counts[format_info] = 1
                
                
        # Sort the format counts by the count in descending order
        format_counts = dict(sorted(format_counts.items(), key=lambda x: x[1], reverse=True))

        # Print statistics
        print("Image Format Statistics:\n")
        print(f"Folder: {folder}")
        for format_info, count in format_counts.items():
            print(f"    - Format: {format_info}, Count: {count}")
        print("Distinct formats:", len(format_counts))
        print("-" * 50, "\n")
        
        
        # Plot the statistics
        plt.figure(figsize=(40, 5))
        plt.bar(format_counts.keys(), format_counts.values())
        
        # rotate the x-axis labels
        plt.xticks(rotation=45)
        plt.xlabel("Image Format")
        plt.ylabel("Count")
        plt.title(f"Image Format Statistics for {folder}")
        
        # show values on top of the bars
        for x, y in zip(format_counts.keys(), format_counts.values()):
            plt.text(x, y, f"{y}", ha='center', va='bottom')  
        
        plt.savefig(f"./readme_images/dataset_images/image_format_statistics_{folder}.png")
        
        plt.show()
   
        
def plot_images_and_padded_images(image_paths : str = None) -> None:
    
    """
    Plot original images and their padded versions side by side.
    
    Args:
    - image_paths (list): List of paths to the input images.
    """
    
    preprocesed_images = [resize_or_pad_image(image_path) for image_path in image_paths]
    
    # Display both original and preprocessed images
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    for i, ax in enumerate(axes[0]):
        # Load and display original image
        original_img = plt.imread(image_paths[i])
        ax.imshow(original_img)
        ax.axis('off')
        ax.set_title('Original Image')
        
    for i, ax in enumerate(axes[1]):
        # Display preprocessed image
        padded_image = transforms.ToPILImage()(preprocesed_images[i].squeeze(0))
        ax.imshow(padded_image) 
        ax.axis('off')
        ax.set_title('Padded / Resized Image')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("./readme_images/dataset_images/original_padded_images.png") 
