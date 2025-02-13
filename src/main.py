import argparse
import yaml
import os
import torch
import random
import numpy as np

from utils.training import Trainer

def get_args():
    parser = argparse.ArgumentParser("Train a model on Wildfire Dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def seed_everything(seed):
    """fix the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    args = get_args()

    print(f"Using config: {args}")
    
    os.environ["HF_HOME"] = "/data/amathur-23/cache/"
    seed_everything(args.seed)
    args.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()