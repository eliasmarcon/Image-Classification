import torch
import logging
import time

from typing import Tuple


class Trainer():
    
    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_loader,
                 train_metric,
                 val_loader,
                 val_metric,
                 device,
                 num_epochs,
                 logger,
                 save_dir,
                 batch_size,
                 val_freq,
                 early_stopping_patience) -> None:
        
        """
        Initialize the Trainer class.

        Args:
        - model (torch.nn.Module): The model to train.
        - optimizer (torch.optim.Optimizer): The optimizer.
        - loss_fn (torch.nn.Module): The loss function.
        - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        - train_loader (DataLoader): The DataLoader for the training dataset.
        - train_metric: The metric to evaluate training performance.
        - val_loader (DataLoader): The DataLoader for the validation dataset.
        - val_metric: The metric to evaluate validation performance.
        - device (torch.device): Device to run the training on.
        - num_epochs (int): Number of epochs to train.
        - logger: Logger to record the results.
        - save_dir (str): Directory to save models.
        - batch_size (int): Batch size for training.
        - val_freq (int): Frequency of validation (in epochs).
        - early_stopping_patience (int): Patience for early stopping.
        """        
        
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.train_metric = train_metric
        self.val_loader = val_loader
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.logger = logger
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.val_freq = val_freq
        self.early_stopping_patience = early_stopping_patience
        
        self.num_train_data = len(train_loader.dataset)
        self.num_val_data = len(val_loader.dataset)
        self.best_val_loss = float('inf')
        self.best_acc = 0.0
        

    def train(self) -> None:
        
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per class accuracy on validation data set is higher
        than currently saved best mean per class accuracy. 
        Depending on the val_freq parameter, validation is not performed every epoch.
        """
        
        # Initialize patience counter for early stopping
        patience_counter = 0
        
        # Training loop
        for epoch_idx in range(1, self.num_epochs + 1):
            
            starting_time = time.time()
            
            # Train for one epoch
            train_loss, train_acc, train_pcacc = self._train_epoch()

            # check if validation should be performed
            if epoch_idx % self.val_freq == 0:

                # validate
                val_loss, val_acc, val_pcacc = self._val_epoch()

                # Check if the current per class accuracy and validation loss is better than the best
                if val_acc > self.best_acc:
                    self.best_acc = val_acc

                    # Save the model as the best per class accuracy model
                    self.model.save(self.save_dir, f"best_val_acc")

                # Check if the current validation loss is better than the best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

                    # Save the model as the best validation loss model
                    self.model.save(self.save_dir, f"best_val_loss")

                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1  # Increment patience counter

                # If the patience counter reaches the threshold, stop training early
                if patience_counter >= self.early_stopping_patience:
                    logging.info(f"Early stopping triggered. Training stopped after {epoch_idx}/{self.num_epochs} epochs.")
                    break
                
            logging.info(f"Epoch {epoch_idx}/{self.num_epochs} completed in {time.time() - starting_time:6.2f} seconds")
            logging.info(f"Train Loss: {train_loss:8.4f} | Train Accuracy: {train_acc:6.4f} | Train Per Class Accuracy: {train_pcacc:6.4f}")
            logging.info(f"Val Loss:   {val_loss:8.4f} | Val Accuracy:   {val_acc:6.4f} | Val Per Class Accuracy:   {val_pcacc:6.4f}\n")
            
            # log metrics and loss using W&B logger
            if self.logger:
                
                self.logger.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_per_class_accuracy": train_pcacc,
                    "val_loss": val_loss if (epoch_idx + 1) % self.val_freq == 0 else None,
                    "val_accuracy": val_acc if (epoch_idx + 1) % self.val_freq == 0 else None,
                    "val_per_class_accuracy": val_pcacc if (epoch_idx + 1) % self.val_freq == 0 else None,
                })
                
        # save the final model
        self.model.save(self.save_dir, f"terminal")
    
    
    def _train_epoch(self) -> Tuple[float, float, float]:
        
        """
        Training logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch.
        """
        
        # Reset the training metric
        self.train_metric.reset()
        
        # Initialize the epoch loss
        epoch_loss = 0.0
        
        # Training loop
        self.model.train()
        
        for inputs, targets in self.train_loader:
            
            # Move inputs and targets to the specified device
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.shape[0]

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
                        
            # Backward pass
            ## Zero the gradients
            self.optimizer.zero_grad()
            ## Calculate the gradients
            loss.backward()
            
            # Adjust learning rate
            self.optimizer.step()

            # Update the loss
            epoch_loss += ( loss.item() * batch_size ) # multiply by batch size to avoid small potential error due to varying batch sizes

            # Update the training metric
            self.train_metric.update(outputs, targets)

        self.lr_scheduler.step()

        # Calculate average loss for the epoch
        epoch_loss /= self.num_train_data
        
        # Calculate training metrics
        acc = self.train_metric.accuracy()
        pcacc = self.train_metric.per_class_accuracy()
        
        return epoch_loss, acc, pcacc
    
    
    def _val_epoch(self) -> Tuple[float, float, float]:
        
        """
        Validation logic for one epoch. 
        Prints current metrics at end of epoch.
        Returns loss, mean accuracy and mean per class accuracy for this epoch on the validation data set.

        epoch_idx (int): Current epoch number
        """
        
        # Reset the validation metric
        self.val_metric.reset()
        
        # Initialize the epoch loss
        epoch_loss = 0.0
        
        # Validation loop
        self.model.eval()
        
        with torch.no_grad():
            
            for inputs, targets in self.val_loader:
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.shape[0]

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # Update the loss
                epoch_loss += ( loss.item() * batch_size ) # multiply by batch size to avoid small potential error due to varying batch sizes

                # Update the validation metric
                self.val_metric.update(outputs, targets)

        # Calculate average loss for the epoch
        epoch_loss /= self.num_val_data
        
       # Calculate validation metrics
        acc = self.val_metric.accuracy()
        pcacc = self.val_metric.per_class_accuracy()
        
        return epoch_loss, acc, pcacc
