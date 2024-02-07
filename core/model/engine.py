import os
import torch
from torch.optim.lr_scheduler import StepLR  # Adjust the import based on the scheduler you want to use

def train_one_epoch(device, model, train_loader, optimizer, criterion, metric, epoch, num_epochs, scheduler=None):
    """
    Trains the model for one epoch.

    Args:
        device (torch.device): The device to be used for training.
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        metric (callable): The metric used for evaluation.
        epoch (int): The current epoch number.
        num_epochs (int): The total number of epochs.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): The learning rate scheduler. Defaults to None.

    Returns:
        tuple: A tuple containing the average loss and average metric value for the epoch.
    """
    model = model.to(device)
    model.train()

    len_dl = len(train_loader)
    epoch_loss, epoch_metric = 0, 0
    
    for step, batch_data in enumerate(train_loader):

        inputs = batch_data["img"].to(device)
        targets = batch_data["seg"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        outputs = outputs.to(device)

        loss = criterion(outputs, targets)
        dice = metric(outputs, targets)

        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_metric += dice.item()

    if scheduler is not None:
        scheduler.step(epoch_loss/len_dl)

    return epoch_loss/len_dl, epoch_metric/len_dl


def test_one_epoch(device, model, test_loader, criterion, metric):
    """
    Test one epoch of the model.

    Args:
        device (torch.device): The device to run the model on.
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for testing.
        criterion (torch.nn.Module): The loss function.
        metric (callable): The evaluation metric.

    Returns:
        tuple: A tuple containing the average loss and average metric value for the epoch.
    """

    len_dl = len(test_loader)
    epoch_loss, epoch_metric = 0, 0
    
    with torch.no_grad():
        model.eval()
        for step, batch_data in enumerate(test_loader):

            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)

            epoch_loss += loss.item()
            epoch_metric += dice.item()

    return epoch_loss/len_dl, epoch_metric/len_dl
