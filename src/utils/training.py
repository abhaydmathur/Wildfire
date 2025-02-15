import torch
from torch.utils.data import DataLoader
from utils.datasets import WildfireDataset
from models import JustCoords, ResNetEncoder, ResNetBinaryClassifier, ConvVAE, ClassifierFeatures
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import f1_score
from torchvision import transforms

from utils.logging_tb import Logger

from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from utils.augmentations import ContrastiveTransformations

from utils.losses import BetaVAELoss



class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        self.train_dataset = WildfireDataset(
            args.data_path, split="train", labeled=True
        )

        self.val_dataset = WildfireDataset(args.data_path, split="val")

        self.test_dataset = WildfireDataset(args.data_path, split="test")

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        self.model_name = args.model_name
        self.epochs = args.epochs
        self.model_save_path = args.model_save_path
        self.log_dir = args.log_dir

        if self.model_name == "resnet_classifier":
            self.model = ResNetBinaryClassifier(
                backbone=args.backbone,
                pretrained=args.pretrained,
                train_backbone=args.train_backbone,
            ).to(self.device)

        elif self.model_name == "just_coords":
            self.model = JustCoords().to(self.device)

        else:
            raise NotImplementedError("Model not implemented")
        

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=args.learning_rate
        )

        self.criterion = torch.nn.BCELoss()

        self.log_path = os.path.join(args.log_dir, self.args.trial_name)
        os.makedirs(self.log_path, exist_ok=True)
        self.logger = Logger(self.log_path)

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_epoch = 0

        self.training_history = {}

    def save_to_log(self, logdir, logger, info, epoch, w_summary=False, model=None):
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

            if self.model_name == "resnet_classifier":
                images = batch["image"].to(self.device)
                inputs = images
            elif self.model_name == "just_coords":
                inputs = coords
            else:
                raise NotImplementedError("Model not implemented")

            self.optimizer.zero_grad()
            outputs = self.model(inputs).squeeze()
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

            if self.model_name == "resnet_classifier":
                images = batch["image"].to(self.device)
                inputs = images
            elif self.model_name == "just_coords":
                inputs = coords
            else:
                raise NotImplementedError("Model not implemented")

            with torch.no_grad():
                outputs = self.model(inputs).squeeze()
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

        f1 = f1_score(label_list, np.round(pred_list))

        info = {
            "loss": np.mean(losses),
            "acc": np.mean(accs),
            "f1_score": f1,
        }
        return info

    def log_init(self):
        train_info = self.validate(split="train")
        val_info = self.validate(split="val")
        test_info = self.validate(split="test")

        log_train_info = {f"train/{k}": v for k, v in train_info.items()}
        log_val_info = {f"val/{k}": v for k, v in val_info.items()}
        log_test_info = {f"test/{k}": v for k, v in test_info.items()}

        self.save_to_log(self.args.log_dir, self.logger, log_train_info, 0)
        self.save_to_log(self.args.log_dir, self.logger, log_val_info, 0)
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, 0)

    def train(self):
        self.log_init()

        for epoch in range(self.epochs):
            epoch_info = self.train_epoch(epoch)

            log_epoch_info = {f"train/{k}": v for k, v in epoch_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_epoch_info, epoch + 1)

            val_info = self.validate()
            log_val_info = {f"val/{k}": v for k, v in val_info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_val_info, epoch + 1)

            self.training_history[epoch] = {
                "train": epoch_info,
                "val": val_info,
            }

            os.makedirs(self.model_save_path, exist_ok=True)

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
                    print("Could not remove latest model")
            self.latest_model_path = os.path.join(
                self.model_save_path, f"latest_model_{epoch+1}eps.pth"
            )
            self.save(self.latest_model_path)

            if self.execute_callbacks(epoch):
                break

            if (epoch + 1) % self.args.test_freq == 0:
                test_info = self.validate(split="test")
                log_test_info = {f"test/{k}": v for k, v in test_info.items()}
                self.save_to_log(
                    self.args.log_dir, self.logger, log_test_info, epoch + 1
                )

        test_info = self.validate(split="test")
        log_test_info = {f"test/{k}": v for k, v in test_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, epoch + 1)

    def execute_callbacks(self, epoch):
        # Early Stopping
        if self.args.early_stopping_patience is not None:
            if epoch - self.best_epoch > self.args.early_stopping_patience:
                print(f"Early Stopping after {epoch} epochs")
                return True
        return False

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def run_tests(self):
        test_info = self.validate(split="test")
        print(test_info)
        log_test_info = {f"test/{k}": v for k, v in test_info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_test_info, 1)


class SimCLRTrainer:

    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.temperature = self.args.temperature
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.transforms = ContrastiveTransformations(img_size=350)

        self.train_dataset = WildfireDataset(
            self.args.data_path,
            split="train",
            labeled=False,
            transforms=self.transforms,
        )

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

        self.model = ResNetEncoder(
            out_features=128,
            pretrained=self.args.pretrained,
            train_backbone=self.args.train_backbone,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1
        )

        self.model = self.model.to(self.device)

        self.log_dir = os.path.join(self.args.log_dir, self.args.trial_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = Logger(self.log_dir)

        self.model_save_path = os.path.join(self.args.model_save_path, self.args.trial_name)
        os.makedirs(self.model_save_path, exist_ok=True)

    def info_nce_loss(self, features):
        # print(features)
        cos_sim = F.cosine_similarity(
            features[:, None, :], features[None, :, :], dim=-1
        )

        # Mask out the positive samples
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        cos_sim = cos_sim / self.temperature
        loss = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        loss = loss.mean()

        return loss

    def validate(self):
        losses = []

        num_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            # print(batch['image'].shape)
            with torch.no_grad():
                images = torch.cat(list(batch["image"]), dim=0)
                images = images.to(self.device)

                with autocast(
                    device_type="cuda", dtype=torch.float16
                ):  # to improve performance while maintaining accuracy.
                    # with autocast():
                    features = self.model(images)
                    loss = self.info_nce_loss(features)

                losses.append(loss.item())

            print(
                f"\rEvaluation ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
                end="",
            )

        print(
            f"\rEvaluation ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
        )

        return {"loss": np.mean(losses)}
    
    def init_logs(self):
        info = self.validate()
        log_info = {f"train/{k}": v for k, v in info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_info, 0)

    def train(self):
    
        self.init_logs()
        self.scaler = GradScaler()  # gradient scaling, useful when we use float16

        print("Start SimCLR training for {} epochs.".format(self.epochs))
        losses = []

        for epoch in range(self.epochs):
            info = self.train_epoch(epoch)
            losses.append(info["loss"])

            log_info = {f"train/{k}": v for k, v in info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_info, epoch + 1)

            if losses[-1] <= np.min(losses):
                print("Saving best model at epoch {}".format(epoch))
                
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_save_path, "simclr_model.pth"
                    ),
                )

            if len(losses) - np.argmin(losses) > self.args.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs")
                break

    def save_to_log(self, logdir, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

    def train_epoch(self, epoch):
        losses = []

        num_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            # print(batch['image'].shape)
            images = torch.cat(list(batch["image"]), dim=0)
            images = images.to(self.device)

            with autocast(
                device_type="cuda", dtype=torch.float16
            ):  # to improve performance while maintaining accuracy.
                # with autocast():
                features = self.model(images)
                loss = self.info_nce_loss(features)

            losses.append(loss.item())

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            print(
                f"\rEpoch: [{epoch}/{self.epochs}] ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
                end="",
            )

        # warmup for the first 10 epochs
        if epoch >= 5:
            self.scheduler.step()

        print(
            f"\rEpoch: [{epoch}/{self.epochs}] ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
        )

        return {"loss": np.mean(losses)}
    
class VAETrainer:
    def __init__(self, args):
        self.args = args
        self.device = self.args.device
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.latent_dim = self.args.latent_dim
        
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])            

        self.train_dataset = WildfireDataset(
            args.data_path, split="train", labeled=False, transforms=self.transforms
        )
        
        self.train_dataset_labeled = WildfireDataset(
            args.data_path, split="train", labeled=True, transforms=self.transforms)

        self.val_dataset = WildfireDataset(args.data_path, split="val", transforms=self.transforms)

        self.test_dataset = WildfireDataset(args.data_path, split="test", transforms=self.transforms)

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Train labeled dataset size: {len(self.train_dataset_labeled)}")
        print(f"Val dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
        )

        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        # VAE model
        self.model = ConvVAE(latent_dim=self.args.latent_dim).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(self.train_loader), eta_min=0, last_epoch=-1
        )

        self.criterion = BetaVAELoss(beta=self.args.beta)
        
        # Classifier model
        self.classifier = ClassifierFeatures(self.model, self.device, input_dim=self.args.latent_dim).to(self.device)
        
        self.classification_optimizer = torch.optim.Adam(
            self.classifier.parameters(), lr=self.args.learning_rate_classifier, weight_decay=self.args.weight_decay
        )
        
        self.classification_criterion = torch.nn.BCELoss()

        self.log_dir = os.path.join(args.log_dir, self.args.trial_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = Logger(self.log_dir)

        self.model_save_path = os.path.join(args.model_save_path, self.args.trial_name)
        os.makedirs(self.model_save_path, exist_ok=True)

        self.best_model_path = None
        self.latest_model_path = None
        self.best_loss = np.inf
        self.best_epoch = 0

        self.training_history = {}
        
    def init_logs(self):
        info = self.validate()
        log_info = {f"train/{k}": v for k, v in info.items()}
        self.save_to_log(self.args.log_dir, self.logger, log_info, 0)
    
    def train(self):
        self.train_vae()
        self.train_classifier()
        self.validate_classifier()
    
    def train_vae(self):
        self.init_logs()
        print("Start VAE training for {} epochs.".format(self.epochs))
        losses = []
        
        for epoch in range(self.epochs):
            info = self.train_epoch(epoch)
            losses.append(info["loss"])
            
            log_info = {f"train/{k}": v for k, v in info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_info, epoch + 1)
            
            if losses[-1] <= np.min(losses):
                print("Saving best model at epoch {}".format(epoch))
                
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_save_path, "vae_model.pth"
                    ),
                )
            if len(losses) - np.argmin(losses) > self.args.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs")
                break
        
    def train_epoch(self, epoch):
        
        losses = []
        n_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            self.optimizer.zero_grad()
            recon_images, mu, logvar = self.model(images)
            loss = self.criterion(images, recon_images, mu, logvar)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            print(
                f"\rEpoch: [{epoch}/{self.epochs}] ({i}/{n_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
                end="",
            )

        # warmup for the first 10 epochs
        if epoch >= 5:
            self.scheduler.step()

        print(
            f"\rEpoch: [{epoch}/{self.epochs}] ({i}/{n_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
        )

        return {"loss": np.mean(losses)}

    def save_to_log(self, logdir, logger, info, epoch, w_summary=False, model=None):
        # save scalars
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

    def validate(self):
        losses = []

        num_steps = len(self.train_loader)
        for i, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            with torch.no_grad():
                recon_images, mu, logvar = self.model(images)
                loss = self.criterion(images, recon_images, mu, logvar)
            losses.append(loss.item())

            print(
                f"\rEvaluation ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
                end="",
            )

        print(
            f"\rEvaluation ({i}/{num_steps}), Average loss: {np.mean(losses):.4f}, lr: {self.scheduler.get_last_lr()[0]:.4f}",
        )

        return {"loss": np.mean(losses)}
    
    def train_classifier(self,):
        self.init_logs()
        print("Start VAE Classifier training for {} epochs.".format(self.epochs_classifier))
        losses = []
        
        for epoch in range(self.epochs_classifier):
            info = self.train_classifier(epoch)
            losses.append(info["loss"])
            
            log_info = {f"train_classfier/{k}": v for k, v in info.items()}
            self.save_to_log(self.args.log_dir, self.logger, log_info, epoch + 1)
            
            if losses[-1] <= np.min(losses):
                print("Saving best model at epoch {}".format(epoch))
                
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.model_save_path, "vae_classifier_model.pth"
                    ),
                )
            if len(losses) - np.argmin(losses) > self.args.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs")
                break
    
    def train_classifier_epoch(self,):
        self.classifier.train()
        correct = 0
        total = 0
        losses = []

        for batch in self.train_dataset_labeled:
            target = batch['label'].float().to(self.device)  # Move labels to the correct device

            self.classification_optimizer.zero_grad()
            output = self.classifier(batch).squeeze()  # Ensure shape consistency

            loss = self.classification_criterion(output, target)
            loss.backward()
            self.classification_optimizer.step()

            losses.append(loss.item())
            predicted = (output > 0.5).float()  # Binary classification threshold
            correct += (predicted == target).sum().item()
            total += target.size(0)

        accuracy = 100. * correct / total
        return np.mean(losses), accuracy
    
    def validate_classifier(self):
        self.classifier.eval()
        losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_dataset:
                target = batch['label'].float().to(self.device)
                output = self.classifier(batch).squeeze()
                loss = self.classification_criterion(output, target)
                losses.append(loss.item())
                predicted = (output > 0.5).float()
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
        accuracy = 100. * correct / total
        return np.mean(losses), accuracy
        
                
        