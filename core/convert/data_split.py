import glob
import json
import tqdm
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

def split_data_and_create_json():
    """
    Split the data into train, validation, and test sets, and create a JSON file containing the split information.

    Returns:
        None
    """
    imgs_list = sorted(glob.glob("data/Images/*.png"))
    segs_list = sorted(glob.glob("data/Contourages/*.png"))

    train_ids, val_ids = train_test_split(range(len(imgs_list)), test_size=0.3, shuffle=True, random_state=12345)
    valid_ids, test_ids = train_test_split(val_ids, test_size=0.333, shuffle=True, random_state=6789)

    out = {
        "class_names": ["background", "lesion"],
        "class_weights": {},
        "train": [],
        "valid": [],
        "test": []
    }

    for k in ["train", "valid", "test"]:
        for id in tqdm.tqdm(eval(f"{k}_ids")):
            out[k].append(
                {
                    "img": imgs_list[id],
                    "seg": segs_list[id],
                }
            )

    num_classes = len(out["class_names"])
    num_samples_class = np.zeros(num_classes)
    for data in tqdm.tqdm(out["train"], total=len(out["train"])):
        file = data["seg"]
        seg = np.asarray(Image.open(file).convert("1"), dtype="uint8")
        for c in range(num_classes):
            num_samples_class[c] += len(seg[seg==c])

    num_samples = seg.shape[0] * seg.shape[1] * len(out["train"]) 
    class_weights = num_samples / (num_classes * num_samples_class)

    for c in range(num_classes):
        out["class_weights"][c] = class_weights[c]
    
    total_weights = sum(out["class_weights"].values())
    for c in range(num_classes):
        out["class_weights"][c] /= total_weights

    json.dump(out, open("data.json", "w"), indent=4)
    