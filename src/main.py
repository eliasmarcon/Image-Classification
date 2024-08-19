import torch
import argparse
import os
import logging

# Own modules
import dataset.augmentation_data as data_source
import utils
import pipeline.config as config 
from pipeline.dataloader import DataLoader
from pipeline.metric import Accuracy
from pipeline.trainer import Trainer
from pipeline.tester import Tester


logging.basicConfig(format='%(message)s')
logging.getLogger().setLevel(logging.INFO)

os.environ["WANDB_SILENT"] = "true"

# Set random seed for reproducibility
torch.manual_seed(100)
torch.cuda.manual_seed(100)


def main(args) -> None:
    
    # Load the model
    model = utils.load_model(args.model_type, args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define a loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    
    # create metrics
    train_metric = Accuracy(classes=3)
    val_metric = Accuracy(classes=3)
    test_metric = Accuracy(classes=3)
    
    # create logger directory
    logger, model_save_dir = utils.create_logger_dir(args.model_type, args.pretrained, args.data_augmentation, args.learning_rate, args.weight_decay, args.gamma)    

    
    ############################## Training ################################
    image_paths, labels = data_source.read_data(size=1, augment = args.data_augmentation)
    data_loader = DataLoader(args.batch_size, image_paths, labels)
    train_loader, val_loader = data_loader.get_dataset(dataset_type = "train")
    
    # define trainer    
    trainer = Trainer(model,
                      optimizer,
                      loss_fn,
                      lr_scheduler,
                      train_loader,
                      train_metric,
                      val_loader,
                      val_metric,
                      device,
                      args.num_epochs,
                      logger,
                      model_save_dir,
                      args.batch_size,
                      args.val_freq,
                      args.early_stopping_patience
                    )
    
    logging.info(f"\nStart training for {logger.run_name}\n")
    trainer.train()
    logging.info("Training completed")
    
    
    ############################## Testing ################################
    test_paths, test_labels = data_source.read_data(base_image_path="./dataset_testing/")
    test_loader = DataLoader(args.batch_size, test_paths, test_labels)
    test_loader = test_loader.get_dataset(dataset_type = "test")

    if "VGG" in args.model_type and args.pretrained:
        # Unfreeze all layers for testing
        model.unfreeze_all()
        
    tester = Tester(model, loss_fn, test_loader, test_metric, logger, args.plots, model_save_dir, device)

    logging.info("\nTesting the model...")
    for model_checkpoint in config.test_checkpoints:
        tester.test(model_checkpoint)
    logging.info("Testing completed\n")
    
    
    ############################## Delete Model Checkpoints ################################
    ###################################### if needed #######################################   
    # logging.info(f"Deleting all model checkpoints.")
    # # Delete all model checkpoints after testing due to their large size
    # for checkpoint in Path(model_save_dir).glob("*.pt"):
    #     checkpoint.unlink()
    
    # logging.info(f"Model checkpoints deleted.")
    
    
    # Finalize the W&B run
    logger.finish()
    
    
    
if __name__ == "__main__":
    
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Training')
    
    # Add an argument for model type
    parser.add_argument('-m', '--model_type', default='VGG', type=str,
                        help='model type to train (default: VGG)')
    
    # Add an arugment if pretrained model is used
    parser.add_argument('-p', '--pretrained', default=False, action="store_true",
                        help='pretrained model to train (default: False)')
    
    # Add an argument for specifying the number of epochs
    parser.add_argument('-e', '--num_epochs', default=30, type=int,
                        help='number of epochs to train the model (default: 30)')
    
    # Add an argument for specifying the batch_size
    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='batch size to train model (default: 32)')
    
    # Add an argument for specifying the learning rate
    parser.add_argument('-l', '--learning_rate', default=0.0001, type=float,
                        help='learning rate to train model (default: 0.0001)')
    
    # Add an argument for specifying the weight decay
    parser.add_argument('-w', '--weight_decay', default=0.0001, type=float,
                        help='weight decay to train model (default: 0.0001)')
    
    # Add an argument for gamma value
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='gamma value for learning rate scheduler (default: 0.99)')
    
    # Add an argument for specifying the val_frequency of model
    parser.add_argument('-f', '--val_freq', default=1, type=int,
                        help='validation frequency to run validation (default: 1)')
   
    # Add an argument for specifying the early stopping patience
    parser.add_argument('-es', '--early_stopping_patience', default=20, type=int,
                        help='early stopping patience for training (default: 20)')
   
    # Add an argument for data augmentation
    parser.add_argument('-a', '--data_augmentation', default=False, action="store_true",
                        help='data augmentation for training data (default: False)')
    
    # Add an argument for confusion matrix and heatmaps
    parser.add_argument('-pl', '--plots', default=False, action="store_true",
                        help='confusion matrix and heatmaps for testing data (default: False)')
    
    # Parse the arguments
    args = parser.parse_args()

    
    main(args)