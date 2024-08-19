import os

from plotting_dataset import plot_dataset_size, reduce_dataset, plot_image_format_statistics, plot_images_and_padded_images
from augmentation_data import create_testing_batch, read_data, create_augmented_data, plot_augmented_images

if __name__ == "__main__":
    
    if not os.path.exists("./readme_images/"):
        os.mkdir("./readme_images/")
        
    if not os.path.exists("./readme_images/dataset_images/"):
        os.mkdir("./readme_images/dataset_images/")
    
    ##############################################################################################################
    # Dataset
    plot_dataset_size("./readme_images/dataset_images/dataset_before_reduction.png")
    
    # After the first plot, the dataset was reduced to 1500 images per class    
    reduce_dataset()
    
    plot_dataset_size("./readme_images/dataset_images/dataset_after_reduction.png")
    
    plot_image_format_statistics()
    
    
    ##############################################################################################################
    # Padding Resize data
    create_testing_batch()
    
    image_paths, _ = read_data(size = 0.1)
    plot_images_and_padded_images(image_paths[:5])    
    
    ##############################################################################################################
    # Augmentation data
    image_paths, labels = read_data(size=1)  # Replace with your data loading function
    create_augmented_data(image_paths, labels)
    
    # Plot the augmented data
    plot_augmented_images()
