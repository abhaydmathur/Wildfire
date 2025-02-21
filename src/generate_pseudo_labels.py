import os
os.environ["TORCH_HOME"] = "/data/amathur-23/ROB313/"

import torch
import torch.nn as nn

from utils.datasets import WildfireDataset
from tqdm import tqdm
from models import ResNetCoordsBinaryClassifier

backbone = 'resnet34'
device = "cuda:1"
model_path = "/data/amathur-23/ROB313/models/coords_et_classifier_trial_resnet34/best_model.pth"

model = ResNetCoordsBinaryClassifier(backbone=backbone)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

unlabelled_train_dataset = WildfireDataset("/data/amathur-23/ROB313", split="train", labeled=False)
unlabelled_loader = torch.utils.data.DataLoader(unlabelled_train_dataset, batch_size=16, shuffle=True)

model.eval()

all_filenames = []
all_coordx = []
all_coordy = []
all_preds = []

err_files = []
err_true = []
err_pred = []

for batch in tqdm(unlabelled_loader):
    images = batch["image"]
    coords = batch["coords"]
    filenames = batch["filename"]
    with torch.no_grad():
        outputs = model(images.to(device), coords.to(device))
    preds = torch.round(outputs).squeeze()

    preds = preds.detach().cpu().numpy()

    all_filenames.extend(filenames)
    all_coordx.extend(coords[:, 0].detach().cpu().numpy())
    all_coordy.extend(coords[:, 1].detach().cpu().numpy())
    all_preds.extend(preds)

string_preds = [
    "wildfire" if pred == 1 else "nowildfire"
    for pred in all_preds
]

pseudo_labelled_data = {
    "filename": all_filenames,
    "coord_x": all_coordx,
    "coord_y": all_coordy,
    "label": string_preds
}

import pandas as pd

df = pd.DataFrame(pseudo_labelled_data)
df.to_csv("/data/amathur-23/ROB313/train_pseudo_labeled.csv", index=False)

import numpy as np

psuedo_acc = np.sum([1 if f"/{pred}/" in filename else 0 for filename, pred in zip(all_filenames, string_preds)]) / len(all_filenames)

print(f"Pseudo Label Accuracy: {psuedo_acc}")