import os
import json
import configparser 
from PIL import Image
import tqdm
import numpy as np
import pandas as pd
from random import sample


import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from monai.networks.nets import UNet
import monai.transforms as mt
from monai.networks.layers import Norm, Act
from monai.visualize.utils import blend_images

from core.data.loader import TuftsDataset
from core.data.augmentation import get_transorms
from core.metrics.metric import MeanDiceScore
from core.metrics.loss import CombinedLoss


def evaluate(device, model, data_loader, criterion, metric):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        device (torch.device): The device to perform the evaluation on.
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the evaluation dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        metric (callable): The metric function used for evaluation.

    Returns:
        tuple: A tuple containing the loss values and dice scores for each batch in the evaluation dataset.
    """
    len_dl = len(data_loader)
    Loss, Dice = [], []
    
    with torch.no_grad():
        model.eval()

        for batch_data in tqdm.tqdm(data_loader, total=len_dl):
        
            inputs = batch_data["img"].to(device)
            targets = batch_data["seg"].to(device).unsqueeze(0)
            
            outputs = model(inputs.unsqueeze(0))
            outputs = nn.Softmax(dim=1)(outputs)

            loss = criterion(outputs, targets)
            dice = metric(outputs, targets)

            Loss.append(loss.cpu().numpy())
            Dice.append(dice.cpu().numpy())
            
        return Loss, Dice

if __name__ == "__main__":
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
        
    config = configparser.ConfigParser()
    config.read("config.ini")
    device = config["TRAIN"]["device"]
    gpu_id = config["TRAIN"]["gpu_id"]
    batch_size = int(config["TRAIN"]["batch_size"])

    model_path = os.path.join("model", "model.pt")
    assert os.path.exists(model_path) == True
    
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
    device = torch.device(device)
    
    class_weights = class_weights.to(device)

    # create datasets
    new_shape = (1024, 2048)

    valid_transform = get_transorms(
        new_shape, 
        num_classes=num_classes
    )

    train_ds = TuftsDataset(jfile["train"], masking=False, transform=valid_transform)
    valid_ds = TuftsDataset(jfile["valid"], masking=False, transform=valid_transform)
    test_ds = TuftsDataset(jfile["test"], masking=False, transform=valid_transform)

    # create dataloaders
    batch_size = batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
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
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to(device)
    print("Model loaded.")

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

    out = {
        "file_name": [],
        "set_name": ["train"]*len(jfile["train"]) + ["valid"]*len(jfile["valid"]) + ["test"]*len(jfile["test"]),
        "loss": train_loss + valid_loss + test_loss,
        "dice": train_dice + valid_dice + test_dice
    }

    for set_name in ["train", "valid", "test"]:
        for i, data in enumerate(jfile[set_name]):
            out["file_name"].append(data["img"])

    df = pd.DataFrame(out)
    df.sort_values(by=["file_name"])
    df.to_csv("./outputs/evaluation_results.csv")
    print("Evaluation results saved.")
    
    # The goal is now to create a image that shows the original image, the ground truth, and the prediction.
    # We will use the first image from the test set.
    # First, we need to get the image and the ground truth in pil format.
    # Then passed it to the predict function.
    
    def predict(img_pil, seg_pil=None, model=model, img_id="0"):
        """
        Perform image segmentation prediction using a given model.

        Args:
            img_pil (PIL.Image.Image): The input image in PIL format.
            seg_pil (PIL.Image.Image, optional): The ground truth segmentation mask in PIL format. Defaults to None.
            model (torch.nn.Module, optional): The segmentation model. Defaults to the global model.
            img_id (str, optional): The ID of the image. Defaults to "0".

        Returns:
            None
        """
        global overlay

        transforms = mt.compose.Compose(
            [
                mt.NormalizeIntensity(
                    nonzero=True, 
                    channel_wise=True
                ),
                mt.Resize(
                    spatial_size=(1024, 2048),
                    mode="bilinear"
                ),
                mt.ToTensor(
                    dtype=torch.float32
                )
            ]
        )

        img = np.asarray(img_pil, dtype=np.float32)
        img = np.expand_dims(img, axis=0)
        img_preproc = transforms(img).to(device)
        pred = model(img_preproc.unsqueeze(0))
        pred = torch.nn.Softmax(dim=1)(pred)
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        
        
        img_resized = img_pil.resize((2048, 1024))
        img_resized = np.asarray(img_resized)/255
        img_resized = np.expand_dims(img_resized, 0)

        seg_resized = seg_pil.resize((2048, 1024))
        seg_resized = np.expand_dims(seg_resized, 0)
        overlay = blend_images(img_resized, pred,cmap='rainbow')
        overlay = blend_images(overlay, seg_resized, cmap='brg')
        # save the image
        overlay = np.transpose(overlay, (1, 2, 0))*255
        overlay_pil = Image.fromarray(overlay.astype(np.uint8))
        overlay_pil.save("./outputs/predictions/overlay_" + img_id + ".png")
        
    preds_dir = "./outputs/predictions"
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
    else:
        for file in os.listdir(preds_dir):
            os.remove(os.path.join(preds_dir, file))

    images = sample(jfile["test"], 5)
    for image in images:
        # Get the image and the ground truth
        img_path = image["img"]
        seg_path = image["seg"]
        image_id = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path).convert("L")
        seg = Image.open(seg_path).convert("1")
        predict(img, seg, model, image_id)
        
    print("Predictions saved.")
    
    