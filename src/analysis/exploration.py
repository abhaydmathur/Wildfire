# %%
from utils.datasets import WildfireDataset, show_image, load_image

# %%
from torchvision import transforms
import torch


x = transforms.CenterCrop(224)

random_img = torch.rand(3, 350, 350)
show_image(random_img)
show_image(x(random_img))

# %%
print(random_img.shape)
print(x(random_img).shape)

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
train = WildfireDataset("/data/amathur-23/ROB313", split="train", labeled=True)
train_unlabeled = WildfireDataset(
    "/data/amathur-23/ROB313", split="train", labeled=False
)
val = WildfireDataset("/data/amathur-23/ROB313", split="val", labeled=True)
test = WildfireDataset("/data/amathur-23/ROB313", split="test", labeled=True)


print("Train:", len(train))
print("Train unlabeled:", len(train_unlabeled))
print("Val:", len(val))
print("Test:", len(test))

# %%
sample = train_unlabeled[0]

# %%
sample.keys()

# %%
img = sample["image"]
show_image(img)

# %%
sample["coords"]

# %%
train.meta

# %%
train.meta["int_label"] = [int(x == "wildfire") for x in train.meta["label"]]
val.meta["int_label"] = [int(x == "wildfire") for x in val.meta["label"]]
test.meta["int_label"] = [int(x == "wildfire") for x in test.meta["label"]]

# %%
train.meta

# %%
import random

# Filter images by label
wildfire_files = [f for (f, l) in zip(train.meta["filename"], train.meta["int_label"]) if l == 1]
no_wildfire_files = [f for (f, l) in zip(train.meta["filename"], train.meta["int_label"]) if l == 0]


# Randomly select two images from each category
selected_wildfire_images = [load_image(img) for img in random.sample(wildfire_files, 2)]
selected_no_wildfire_images = [load_image(img) for img in random.sample(no_wildfire_files, 2)]

# Combine selected images and labels
selected_images = selected_wildfire_images + selected_no_wildfire_images
selected_labels = ["wildfire"] * 2 + ["no_wildfire"] * 2

# Plot the images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, img, label in zip(axes, selected_images, selected_labels):
    show_image(img, ax=ax, title=label)
    # ax.set_title(label)

plt.tight_layout()
plt.show()

# %%
train.meta.corr(numeric_only=True)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

datasets = [train, val, test]
titles = ['Train', 'Validation', 'Test']

for ax, dataset, title in zip(axes, datasets, titles):
    scatter = ax.scatter(
        dataset.meta['coord_x'], 
        dataset.meta['coord_y'], 
        c=dataset.meta['int_label'], 
        cmap='viridis', 
        alpha=0.6
    )
    ax.set_title(title)
    ax.set_xlabel('coord_x')
    ax.set_ylabel('coord_y')

handles, labels = scatter.legend_elements()
# print(labels)
labels = ["no wildfire" if label == '$\\mathdefault{0}$' else "wildfire" for label in labels]
fig.legend(handles, labels, title='Labels', loc='center')
plt.tight_layout()
plt.show()

# %%
np.unique(test.meta['int_label'], return_counts=True)

# %%
scatter.legend_elements()

# %%
from torch.utils.data import DataLoader

train_loader = DataLoader(train, batch_size=32, shuffle=True)
val_loader = DataLoader(val, batch_size=32, shuffle=False)
test_loader = DataLoader(test, batch_size=32, shuffle=False)
train_unlabeled_loader = DataLoader(train_unlabeled, batch_size=32, shuffle=True)

# %%
# from tqdm import tqdm

# # Initialize lists to store mean and std values for each channel
# mean_r, mean_g, mean_b = [], [], []
# std_r, std_g, std_b = [], [], []

# # Iterate through the dataset
# for sample in tqdm(train_unlabeled):
#     img = sample["image"]
#     mean_r.append(img[0].mean().item())
#     mean_g.append(img[1].mean().item())
#     mean_b.append(img[2].mean().item())
#     std_r.append(img[0].std().item())
#     std_g.append(img[1].std().item())
#     std_b.append(img[2].std().item())

# # Compute the overall mean and std for each channel
# mean_r = np.mean(mean_r)
# mean_g = np.mean(mean_g)
# mean_b = np.mean(mean_b)
# std_r = np.mean(std_r)
# std_g = np.mean(std_g)
# std_b = np.mean(std_b)

# print(f"Mean R: {mean_r}, Mean G: {mean_g}, Mean B: {mean_b}")
# print(f"Std R: {std_r}, Std G: {std_g}, Std B: {std_b}")

# %%
# mean_r = np.mean(mean_r)
# mean_g = np.mean(mean_g)
# mean_b = np.mean(mean_b)
# std_r = np.mean(std_r)
# std_g = np.mean(std_g)
# std_b = np.mean(std_b)

# print(f"Mean R: {mean_r}, Mean G: {mean_g}, Mean B: {mean_b}")
# print(f"Std R: {std_r}, Std G: {std_g}, Std B: {std_b}")

# %% [markdown]
# 

# %%
# from tqdm import tqdm

# for batch in tqdm(train_loader):
#     continue

# for batch in tqdm(val_loader):
#     continue

# for batch in tqdm(test_loader):
#     continue

# for batch in tqdm(train_unlabeled_loader):
#     continue

# %%
from collections import Counter

train_labels_count = Counter(train.meta["label"])
val_labels_count = Counter(val.meta["label"])
test_labels_count = Counter(test.meta["label"])

print("Train labels count:", train_labels_count)
print("Val labels count:", val_labels_count)
print("Test labels count:", test_labels_count)

# %%
from utils.augmentations import ContrastiveTransformations

clr_transforms = ContrastiveTransformations(img_size=350)

clr_set = WildfireDataset(
    "/data/amathur-23/ROB313", split="train", labeled=True, transforms=clr_transforms
)

# %%
i = 0

# %%
i += 1
sample = clr_set[i]
im1, im2 = sample["image"]

show_image(im1)
show_image(im2)

# %%
import torch.nn as nn

encoder = nn.Sequential(
    nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 350 -> 175
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 175 -> 86
    nn.ReLU(),
    nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 86 -> 43
    nn.ReLU(),
    nn.Conv2d(128, 128, 4, stride=2, padding=1),  # 43 -> 21
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(128 * 21 * 21, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

encoder.eval()

encoder(im1.unsqueeze(0)).shape

# %%
128 * 21 * 21

# %%
conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1)  # 350 -> 175
conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  # 175 -> 86
conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 86 -> 43
conv4 = nn.Conv2d(128, 128, 4, stride=2, padding=1)  # 43 -> 21
relu = nn.ReLU()

fc = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 21 * 21, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

x = im1.unsqueeze(0)


x1 = relu(conv1(x))
x2 = relu(conv2(x1))
x3 = relu(conv3(x2))
x4 = relu(conv4(x3))

print(x1.shape)
print(x2.shape)
print(x3.shape)
print(x4.shape)
x = x4 + x3  # Skip connection


x.shape

# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# %%
import os 

os.environ["TORCH_HOME"] = "/data/amathur-23/ROB313"

backbones = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
]

from models import (
    JustCoords,
    ResNetEncoder,
    ResNetBinaryClassifier,
    ResNetCoordsBinaryClassifier,
    BinaryClassifierWithPretrainedEncoder,
    ConvVAE,
    ClassifierFeatures,
    CNNBinaryClassifier,
    CNNBinaryClassifierWithCoords,
)

# Count Params in each model

models = {
    "just_coords": JustCoords(),
    "cnn_binary_classifier": CNNBinaryClassifier(),
    "cnn_binary_classifier_with_coords": CNNBinaryClassifierWithCoords(),
}

for backbone in backbones:
    models[f"{backbone}_binary_classifier"] = ResNetBinaryClassifier(backbone)
    models[f"{backbone}_coords_binary_classifier"] = ResNetCoordsBinaryClassifier(
        backbone
    )
    models[f"{backbone}_pretrained_encoder"] = ResNetEncoder(out_features=128, backbone=backbone)
    models[f"{backbone}_classifier_on_pretrained"] = BinaryClassifierWithPretrainedEncoder(encoder=models[f"{backbone}_pretrained_encoder"])

for name, model in models.items():
    print(name, count_parameters(model))
    



# %%


# %%



