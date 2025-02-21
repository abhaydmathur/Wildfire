# %%
import os
os.environ["TORCH_HOME"] = "/data/amathur-23/ROB313/"

# %%
import torch
import torch.nn as nn

from utils.datasets import WildfireDataset, show_image, load_image
from models import ResNetCoordsBinaryClassifier

backbone = 'resnet34'
device = "cuda:1"
model_path = "/data/amathur-23/ROB313/models/coords_et_classifier_trial_resnet34/best_model.pth"

model = ResNetCoordsBinaryClassifier(backbone=backbone)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# %%
dataset = WildfireDataset("/data/amathur-23/ROB313", split="test")
loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

all_preds = []
all_labels = []

model.eval()

mis_filenames = []
mis_preds = []
mis_labels = []

for batch in tqdm(loader):
    images = batch["image"]
    coords = batch["coords"]
    labels = batch["label"]
    with torch.no_grad():
        outputs = model(images.to(device), coords.to(device))
    preds = torch.round(outputs).squeeze()

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    all_preds.extend(preds)
    all_labels.extend(labels)

    # Find indices where prediction does not match the label
    mismatch_indices = (preds != labels).nonzero()[0]
    for i in mismatch_indices:
        mis_filenames.append(batch["filename"][i])
        mis_preds.append(preds[i])
        mis_labels.append(labels[i])    

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# %%
from sklearn.metrics import f1_score

accuracy = np.sum(np.array(all_labels) == np.array(all_preds)) / len(all_labels)
print(f"Accuracy: {accuracy}")

f1 = f1_score(all_labels, all_preds)
print(f"F1 Score: {f1}")

# %%
false_positives = [mis_filenames[i] for i in range(len(mis_filenames)) if mis_preds[i] == 1 and mis_labels[i] == 0]
false_negatives = [mis_filenames[i] for i in range(len(mis_filenames)) if mis_preds[i] == 0 and mis_labels[i] == 1]

print("False Positives:", false_positives)
print("False Negatives:", false_negatives)

# %%
import random

# Randomly select 4 files from false positives and false negatives
selected_false_positives = random.sample(false_positives, 4)
selected_false_negatives = random.sample(false_negatives, 4)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Display false positives
for i, fp in enumerate(selected_false_positives):
    image = load_image(fp)
    show_image(image, ax=axes[0, i])
    if i == 0:
        axes[0, i].axis('on')
        axes[0, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[0, i].set_ylabel('False Positives', fontsize=16)

# Display false negatives
for i, fn in enumerate(selected_false_negatives):
    image = load_image(fn)
    show_image(image, ax=axes[1, i])
    if i == 0:
        axes[1, i].axis('on')
        axes[1, i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        axes[1, i].set_ylabel('False Negatives', fontsize=16)

plt.tight_layout()
plt.show()

# %%



