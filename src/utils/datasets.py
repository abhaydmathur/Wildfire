import glob
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class WildfireDataset(Dataset):
    def __init__(
            self,
            root_dir,
            split='train',
    ):
        pass

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        return np.array(image)

    def __len__(self):
        pass    

    def __getitem__(self, index):
        pass

