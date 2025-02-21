import glob
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")
    plt.show()

def load_image(path):
    try:
        image = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {path}")
        image = np.random.randint(0, 255, (350, 350, 3))
    image = np.array(image, dtype=np.float32) / 255.0
    return torch.tensor(image).permute(2, 0, 1)


class WildfireDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        labeled=False,
        transforms=None,
        use_pseudo_labels=False,
    ):
        self.root_dir = root_dir
        self.split = split
        if not labeled and split == "train":
            if not use_pseudo_labels:
                self.split = self.split + "_unlabeled"
            else:
                self.split = self.split + "_pseudo_labeled"

        self.labeled = labeled
        if self.split != "train_unlabeled":
            self.labeled = True

        meta_file = f"{root_dir}/{self.split}.csv"

        print(f"Loading meta file: {meta_file}")

        self.meta = pd.read_csv(meta_file)
        self.meta = self.meta.dropna().drop_duplicates()

        ids = np.arange(len(self.meta))
        self.id_to_filename = dict(zip(ids, self.meta["filename"].values))
        if self.labeled:
            self.id_to_label = dict(zip(ids, self.meta["label"].values))
        self.id_to_coordx = dict(zip(ids, self.meta["coord_x"].values))
        self.id_to_coordy = dict(zip(ids, self.meta["coord_y"].values))

        self.transforms = transforms

    def load_image(self, path):
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {path}")
            image = np.random.randint(0, 255, (350, 350, 3))
        image = np.array(image, dtype=np.float32) / 255.0
        return torch.tensor(image).permute(2, 0, 1)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        filename = self.id_to_filename[idx]
        image = self.load_image(filename)
        coords = torch.tensor(
            [self.id_to_coordx[idx], self.id_to_coordy[idx]], dtype=torch.float32
        )

        if self.transforms:
            image = self.transforms(image)

        if self.split == "train_unlabeled":
            return {
                "image": image,
                "coords": coords,
                "filename" : filename
            }
        else:
            label = float(1 if self.id_to_label[idx] == "wildfire" else 0)
            return {
                "image": image,
                "label": torch.tensor(label, dtype=torch.float),
                "coords": coords,
                "filename" : filename
            }
