import torch
import matplotlib.pyplot as plt
import os

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


class Tester():
    
    def __init__(self, model, loss_fn, test_loader, test_metric, logger, plots, save_dir, device) -> None:
        
        """
        Initialize the Tester class.

        Args:
        - model (torch.nn.Module): The model to test.
        - loss_fn (torch.nn.Module): The loss function.
        - test_loader (DataLoader): The DataLoader for the test dataset.
        - test_metric: The metric to evaluate the test performance.
        - logger: Logger to record the results.
        - plots (bool): Whether to generate plots.
        - save_dir (str): Directory to save results.
        - device (torch.device): Device to run the test on.
        """
        
        self.model = model
        self.loss_fn = loss_fn
        self.test_loader = test_loader
        self.test_metric = test_metric
        self.logger = logger
        self.plots = plots
        self.save_dir = save_dir
        self.device = device

        self.num_train_data = len(self.test_loader.dataset)
        
    
    def test(self, test_checkpoint_type) -> None:
        
        """
        Test the model on the test data set and return the loss, mean accuracy and mean per class accuracy.
        
        test_loader: The test data set to test the model on.
        """

        # Load the model checkpoint
        self.model.load(os.path.join(self.save_dir, test_checkpoint_type))
        self.test_checkpoint_type = test_checkpoint_type
        
        # Initialize the test loss, true and predicted labels, and the test metric
        self.test_metric.reset()
        test_loss = 0.0
        y_true = []
        y_pred = []
        
        # Test loop
        self.model.eval()
        
        with torch.no_grad():
        
            for inputs, targets in self.test_loader:

                inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                batch_size = inputs.shape[0]
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                        
                # Update the loss
                test_loss += ( loss.item() * batch_size ) # multiply by batch size to avoid small potential error due to varying batch sizes
                
                # Update the test metric
                self.test_metric.update(outputs, targets)
                
                # Append the true and predicted labels
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        # Calculate average loss for the test set
        test_loss /= self.num_train_data
        
        # Calculate test metrics
        acc = self.test_metric.accuracy()
        pcacc = self.test_metric.per_class_accuracy()
        
        # Log test metrics
        if self.logger:
            
            self.logger.log({
                f"test_loss_{self.test_checkpoint_type}": test_loss,
                f"test_accuracy_{self.test_checkpoint_type}": acc,
                f"test_per_class_accuracy_{self.test_checkpoint_type}": pcacc
            })        
            
        if self.plots:
            # Create confusion matrix
            self._create_confusion_matrix(y_true, y_pred, acc)
            
            # Create heatmaps
            self._create_test_graphics()
            
            
    def _create_test_graphics(self) -> None:
        
        """
        Test the model on the test data set and return the loss, mean accuracy and mean per class accuracy.
        
        test_loader: The test data set to test the model on.
        """
                
        # Storage for collected images
        images = []
        heatmaps = []
        cam_images = []
        cam_prediction = []
        targets_saving = []
    
        if self.logger.run_name.split('_')[0] == "VGG" or self.logger.run_name.split('_')[0] == "VGGAdapted":
            layer = 'base_model.features.34'
        
        elif self.logger.run_name.split('_')[0] == "CustomVGG":
            layer = 'custom_conv_layers.8'
        
        elif self.logger.run_name.split('_')[0] == "ResNet":
            layer = 'layer4.1'
        
        elif self.logger.run_name.split('_')[0] == "CNNBasic":
            layer = 'conv4_2'
        
        else:
            raise NotImplementedError(f"Model type {self.logger.run_name.split('_')[0]} is not implemented, please choose from VGG, VGGAdapted, CustomVGG, ResNet, CNNBasic.")
            
            
        cam_extractor = SmoothGradCAMpp(self.model, target_layer=layer)
        
        for inputs, targets in self.test_loader:

            inputs, targets = inputs.to(self.device), targets.to(self.device).long()
                                
            if len(targets_saving) < 9:  # 3 images per category (0, 1, 2)             
                
                for input_image, target in zip(inputs, targets):
                    
                    target_item = target.item()
                    
                    if targets_saving.count(target_item) < 3:
                    
                        # Forward pass
                        out = self.model(input_image.unsqueeze(0))
                        
                        # Get the predicted class
                        _, predicted = torch.max(out, 1)
                        
                        # Create activation map (tensor class, tensor)
                        activation_map = cam_extractor(predicted.item(), out)
                        
                        # Denormalize the image
                        denorm_image = self._denormalize(input_image.squeeze(0))
                        
                        # Ensure the activation map has only one channel and no batch dimension
                        heatmap = activation_map[0].squeeze(0)
                                                        
                        # Create CAM image by overlaying heatmap on the original image
                        cam_image = overlay_mask(to_pil_image(denorm_image), to_pil_image(heatmap, mode='F'), alpha=0.5)

                        # append the target, image, heatmap and cam image to the lists
                        targets_saving.append(target_item)
                        images.append(denorm_image.cpu().permute(1, 2, 0).numpy())
                        heatmaps.append(heatmap.cpu().numpy())
                        cam_images.append(cam_image)
                        cam_prediction.append(predicted.item())
                                  
        # Once finished, clear the hooks on your model
        cam_extractor.remove_hooks()         
                                    
        # Create heatmaps
        self._create_heatmaps(images, targets_saving, heatmaps, cam_images, cam_prediction)
        
    
    def _create_confusion_matrix(self, y_true, y_pred, acc) -> None:    
        
        """
        Create and save a confusion matrix.

        Args:
        - y_true (list[int]): True labels.
        - y_pred (list[int]): Predicted labels.
        - acc (float): Accuracy of the model.
        """
        
        # Get Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels= ['Axe', 'Book', 'Hammer'])
        disp.plot()
        plt.title(f"Model: {self.logger.run_name.split('_')[0]} | Pretrained: {self.logger.run_name.split('_')[2]} | Test Accuracy: {acc:.2f}%")

        # create model save directory
        cm_save_dir = Path(f"{self.save_dir}/confusion_matrices")
        cm_save_dir.mkdir(exist_ok=True)
        plt.savefig(f"./{cm_save_dir}/cm_{self.test_checkpoint_type.split('.')[0]}.png")
        plt.close()
        
        
    def _create_heatmaps(self, images, targets_saving, heatmaps, cam_images, cam_prediction) -> None:
        
        """
        Create and save heatmaps for each target class.

        Args:
        - images (list[np.ndarray]): List of original images.
        - targets_saving (list[int]): List of targets to save.
        - heatmaps (list[np.ndarray]): List of heatmaps.
        - cam_images (list[Image.Image]): List of CAM images.
        - cam_prediction (list[int]): List of CAM predictions.
        """
        
        mapping = {0 : 'Axe', 1 : 'Book', 2 : 'Hammer'}
        
        # create model save directory
        heatmaps_save_dir = Path(f"{self.save_dir}/heatmaps")
        heatmaps_save_dir.mkdir(exist_ok=True)
        
        subfilter = Path(f"{heatmaps_save_dir}/{self.test_checkpoint_type.split('.')[0]}")
        subfilter.mkdir(exist_ok=True)
        
        # Create a plot with the original image, the heatmap and the cam image for each target
        for index, label in enumerate(targets_saving):
            
            _, axs = plt.subplots(1, 3, figsize=(10, 10))
            
            axs[0].imshow(images[index])
            axs[0].title.set_text("Original Image with GT: " + mapping[label])
            axs[0].axis('off')
            
            axs[1].imshow(heatmaps[index])
            axs[1].title.set_text("Heatmap")
            axs[1].axis('off')
            
            axs[2].imshow(cam_images[index])
            axs[2].title.set_text("CAM Image with prediction: " + mapping[cam_prediction[index]])
            axs[2].axis('off')
        
            plt.tight_layout()
            plt.savefig(f"./{heatmaps_save_dir}/{self.test_checkpoint_type.split('.')[0]}/{self.logger.run_name.split('_')[0]}_{mapping[label]}_{index}.png")
            plt.close()
        

    def _denormalize(self, tensor) -> None:
        
        """
        Denormalize an image tensor.

        Args:
        - tensor (torch.Tensor): Normalized image tensor.

        Returns:
        - torch.Tensor: Denormalized image tensor.
        """
        
        # Define normalization parameters
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
            
        return torch.clamp(tensor, 0, 1)
    