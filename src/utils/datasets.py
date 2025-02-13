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


class WildfireDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        labeled=False,
    ):
        self.root_dir = root_dir
        self.split = split
        if not labeled and split=="train":
            self.split = self.split + "_unlabeled"

        self.labeled = labeled

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

    def load_image(self, path):
        image = Image.open(path).convert("RGB")
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

        if self.labeled:
            label = 1 if self.id_to_label[idx] == "wildfire" else 0
            return {
                "image": image,
                "label": torch.tensor(label, dtype=torch.long),
                "coords": coords,
            }
        else:
            return {
                "image": image,
                "coords": coords,
            }
