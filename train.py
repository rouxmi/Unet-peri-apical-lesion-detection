import os
import configparser
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from evaluation import evaluate

from monai.networks.nets import UNet
from monai.networks.layers import Norm, Act

from core.data.loader import TuftsDataset
from core.data.augmentation import get_transorms
from core.metrics.metric import MeanDiceScore
from core.metrics.loss import  CombinedLoss
from core.model.engine import train_one_epoch, test_one_epoch

def train(device, model, train_loader, valid_loader, optimizer, criterion, metric, num_epochs, max_patience=10, checkpoint_dir="./checkpoint/"):
    """
    Trains a model using the provided data loaders and optimization parameters.

    Args:
        device (torch.device): The device to be used for training (e.g. "cuda" or "cpu").
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        valid_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        criterion (torch.nn.Module): The loss function to be used for training.
        metric (callable): The metric function to be used for evaluation.
        num_epochs (int): The number of training epochs.
        max_patience (int, optional): The maximum number of epochs to wait for improvement in validation loss before early stopping. Defaults to 10.
        checkpoint_dir (str, optional): The directory to save model checkpoints. Defaults to "./checkpoint/".
    """
    # Rest of the code...
def train(device, model, train_loader, valid_loader, optimizer, criterion, metric, num_epochs, max_patience=10, checkpoint_dir="./checkpoint/"):

    if os.path.exists(checkpoint_dir) == False:
        os.makedirs(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, "model.pt")

    history = {
        "train": {
            "loss": [], "dice": []
        },
        "valid": {
            "loss": [], "dice": []
        },
    }

    dict_to_save = {
        "epoch": 0,
        "model_state_dict": None,
        "optimizer_state_dict": None,
        "history": None
    }
    best_loss = torch.inf
    patience = 0
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    for epoch in range(1, num_epochs+1):
        
        train_loss, train_dice = train_one_epoch(device, model, train_loader, optimizer, criterion, metric, epoch, num_epochs, scheduler=scheduler)
        valid_loss, valid_dice = test_one_epoch(device, model, valid_loader, criterion, metric)

        # update history
        history["train"]["loss"].append(train_loss)
        history["train"]["dice"].append(train_dice)
        history["valid"]["loss"].append(valid_loss)
        history["valid"]["dice"].append(valid_dice)
        
        # if loss is inf after epoch 2, stop training
        if epoch > 2 and np.isnan(train_loss):
            print("Training stopped because loss is NaN.")
            break

        # checkpoint 
        if valid_loss < best_loss:
            
            # reset patience
            patience = 0

            # save the new best model and optimizer parameters
            dict_to_save["epoch"] = epoch
            dict_to_save["model_state_dict"] = model.state_dict()
            dict_to_save["optimizer_state_dict"] = optimizer.state_dict()
            dict_to_save["history"] = history
            torch.save(dict_to_save, model_path)

            # update best loss
            print(f"@ epoch {epoch} val_loss decreased from {best_loss:.4f} to {valid_loss:.4f}. Model saved in {model_path}.\n")
            best_loss = valid_loss
        else:
            patience += 1
            print(f"@ epoch {epoch} val_loss did not decrease from {best_loss:.4f}. {patience} epochs of patience.\n")

            if patience == max_patience:
                print(f"val_loss did not decrease for {max_patience} consecutive epochs.")
                print("Model training has stopped!")
                break

    dict_to_save["history"] = history
    torch.save(dict_to_save, model_path)
    
    # A plot of the training and validation loss and dice
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train"]["loss"], label="train")
    plt.plot(history["valid"]["loss"], label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train"]["dice"], label="train")
    plt.plot(history["valid"]["dice"], label="valid")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.savefig("outputs/history.png")
    plt.close()

if __name__ == "__main__":
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    config = configparser.ConfigParser()
    config.read("config.ini")

    device = config["TRAIN"]["device"]
    gpu_id = config["TRAIN"]["gpu_id"]
    batch_size = int(config["TRAIN"]["batch_size"])
    lr = float(config["TRAIN"]["lr"])
    num_epochs = int(config["TRAIN"]["num_epochs"])

    # load data file
    jfile = json.load(open("data.json"))
    class_names = jfile["class_names"]
    num_classes = len(class_names)
    class_weights = torch.tensor(list(jfile["class_weights"].values()), dtype=torch.float32)
    

    # set device
    if device == "cuda" and torch.cuda.is_available():
        device = device + ":" + str(gpu_id)
    else:
        device = "cpu"
    print(f"Using {device} device.")
    
    class_weights = class_weights.to(device)

    # create datasets
    new_shape = (1024, 2048)
    bright_range = (0.8, 1.2)
    rotation_range = (-np.pi/36, np.pi/36)
    scale_range = (0.8, 1.2)
    train_transform = get_transorms(
        new_shape, 
        bright_range=bright_range, 
        rotation_range=rotation_range, 
        scale_range=scale_range, 
        num_classes=num_classes
    )
    valid_transform = get_transorms(
        new_shape, 
        num_classes=num_classes
    )

    train_ds = TuftsDataset(jfile["train"], masking=False, transform=train_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=False, transform=valid_transform)
    test_ds = TuftsDataset(jfile["test"], masking=False, transform=valid_transform)
    # create dataloaders
    batch_size = batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # build model
    model = UNet(
        spatial_dims = 2,
        in_channels = 1,
        out_channels = num_classes,
        channels = (32, 64, 128, 256, 512, 1024, 1024),
        strides = (2, 2, 2, 2, 2, 2),
        num_res_units = 3,
        norm = Norm.INSTANCE_NVFUSER,
        act = Act.SOFTMAX
    ).to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        lr = lr
    )

    # loss function and dice metric
    metric = MeanDiceScore(softmax=True, weights=class_weights)
    criterion = CombinedLoss(dice_weight=3.0, ce_weight=2.0, softmax=True, weights=class_weights, epsilon=1.e-5)
    
    
    print(f"Setting learning rate to {lr:.4e}\n")
    
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        lr = lr
    )

    # train model
    print(f"Training a model to segment {num_classes} classes:\n{class_names}\n")

    train(
        device, 
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        criterion, 
        metric, 
        num_epochs,
        max_patience=10, 
        checkpoint_dir="./model/"
    )   

    # loss function and dice metric
    metric = MeanDiceScore(softmax=False, weights=None, epsilon=0.)
    criterion = CombinedLoss(softmax=False, weights=class_weights)

    # evaluate model
    print(f"Evaluating a model over the training, validation, and test dataset:\n")

    train_loss, train_dice = evaluate(device, model, train_ds, criterion, metric)
    valid_loss, valid_dice = evaluate(device, model, valid_ds, criterion, metric)
    test_loss, test_dice = evaluate(device, model, test_ds, criterion, metric)

    print(len(train_loss), len(train_dice))

    print(f"Training: {np.mean(train_loss, 0):.4f} loss, {np.nanmean(train_dice, 0):.4f} dice.")
    print(f"Validation: {np.mean(valid_loss, 0):.4f} loss, {np.nanmean(valid_dice, 0):.4f} dice.")
    print(f"Test: {np.mean(test_loss, 0):.4f} loss, {np.nanmean(test_dice, 0):.4f} dice.")
