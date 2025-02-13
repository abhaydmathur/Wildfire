import torch
from torch.utils.data import DataLoader
from utils.datasets import WildfireDataset
from models import JustCoords
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import f1_score

from utils.logging import Logger

class Trainer():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.model = JustCoords().to(self.device)

        self.train_dataset = WildfireDataset(
            args.data_path,
            split="train",
            labeled=True
        )

        self.val_dataset = WildfireDataset(
            args.data_path,
            split="val"
        )

        self.test_dataset = WildfireDataset(
            args.data_path,
            split="test"
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate
        )

        self.criterion = torch.nn.BCELoss()

        self.model_name = args.model_name
        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.model_save_path = args.model_save_path
        self.log_dir = args.log_dir

        self.log_path = os.path.join(args.log_dir, self.model_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = Logger(self.log_path)

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_epoch = 0

        self.training_history = {}
   
    def save_to_log(self, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        # save summaries of weights and biases
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                try:
                    logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                except:
                    continue
                    logger.histo_summary(tag, value.data, epoch)
                if value.grad is not None:
                    logger.histo_summary(
                        tag + "/grad", value.grad.data.cpu().numpy(), epoch
                    )
    
    def train_epoch(self, epoch, verbose=True):
        self.model.train()
        losses = []
        accs = []

        n_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            coords = batch["coords"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(coords)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            with torch.no_grad():
                acc = (outputs.round() == labels).float().mean().item()
                accs.append(acc)
            
            if verbose:
                print(
                    f"\rEpoch {epoch + 1} [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ",
                    end="",
                )
        print()

        info = {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return info
        
    def validate(self, verbose=True, split="val"):
        if split == "val":
            loader = self.val_loader
        elif split == "train":
            loader = self.train_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError("Invalid split")
        self.model.eval()
        losses = []
        accs = []
        label_list = []
        pred_list = []

        n_steps = len(loader)
        for i, batch in enumerate(loader):
            coords = batch["coords"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.no_grad():
                outputs = self.model(coords)
                loss = self.criterion(outputs, labels)
                        

            losses.append(loss.item())
            with torch.no_grad():
                acc = (outputs.round() == labels).float().mean().item()
                accs.append(acc)

            label_list.extend(labels.cpu().numpy())
            pred_list.extend(outputs.cpu().numpy())
            
            if verbose:
                print(
                    f"\rEvaluation on {split} : [{i + 1}/{n_steps}] loss: {np.mean(losses):.3f}, acc: {np.mean(accs):.3f}   ",
                    end="",
                )
        print()

        f1_score = f1_score(label_list, np.round(pred_list))

        info = {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "f1_score": f1_score,
        }
        return info
    
    def train(self):
        self.log_init()

        for epoch in range(self.epochs):
            epoch_info = self.train_epoch()

            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch + 1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch + 1)

            self.training_history[epoch] = {
                "train": epoch_info,
                "val": val_info,
            }

            # Save best and latest models
            if (
                self.best_model_path is None or val_info["loss"] < self.best_loss
            ):  # TODO @abhaydmathur : best metrics??
                self.best_loss = val_info["loss"]
                self.best_model_path = os.path.join(
                    self.model_save_path, "best_model.pth"
                )
                self.save(self.best_model_path)
                self.best_epoch = epoch

            if self.latest_model_path is not None:
                try:
                    os.remove(self.latest_model_path)
                except:
                    try:
                        self.model.remove(self.latest_model_path)
                    except:
                        print(".")
                    print("Could not remove latest model")
            self.latest_model_path = os.path.join(
                self.model_save_path, f"latest_model_{epoch+1}eps.pth"
            )
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break

            if (epoch + 1) % self.args.test_freq == 0:
                test_info = self.validate(split="test")
                for loader_name in test_info.keys():
                    log_test_info = {f"test/{loader_name}/{k}": v for k, v in test_info[loader_name].items()}
                    self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch+1)

        test_info = self.validate(split='test')
        log_test_info = {f"test/{k}": v for k, v in test_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch+1)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False

    def run_tests(self):
        test_info = self.validate(split="test")
        print(test_info)
        log_test_info = {f"test/{k}": v for k, v in test_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, 1)
