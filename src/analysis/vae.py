# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import copy
import os
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset

# %%
from utils.datasets import WildfireDataset

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = WildfireDataset(
    "/data/amathur-23/ROB313", split="train", labeled=False, transforms=transform
)
data_train_labeled = WildfireDataset(
    "/data/amathur-23/ROB313", split="train", labeled=True, transforms=transform
)
val_dataset = WildfireDataset(
    "/data/amathur-23/ROB313", split="val", transforms=transform
)
test_dataset = WildfireDataset(
    "/data/amathur-23/ROB313", split="test", transforms=transform
)

# %%
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader_labeled = DataLoader(
    data_train_labeled, batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()

        # Encoder
        # 3x224x224
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
        )

        self.encoder_output_dim = 256 * 7 * 7
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.Linear(256, self.encoder_output_dim)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = self.decode(z)
        return z, mu, log_var

# %%
class BetaVAELoss(nn.Module):
    def __init__(self, beta=1):
        super(BetaVAELoss, self).__init__()
        self.beta = beta

    def forward(self, x, recon_x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss


criterion_vae = BetaVAELoss(beta=1)

# %%
from tqdm import tqdm


def train(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, f"Training {epoch}"):
        data = batch["image"].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = criterion_vae(data, recon_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader.dataset)


# Validation Function
def validate(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, f"Validation {epoch}"):
            data = batch["image"].to(device)
            recon_batch, mu, logvar = model(data)
            loss = criterion_vae(data, recon_batch, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(dataloader.dataset)

# %%
latent_dim = 256
learning_rate = 1e-5
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvVAE(latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device, epoch)
    print(f"Epoch {epoch} Train loss: {train_loss}")
    val_loss = validate(model, val_loader, device, epoch)
    print(f"Epoch {epoch} Validation loss: {val_loss}")

# %%
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


def perform_clustering(features, method="kmeans", num_clusters=5):
    if method == "kmeans":
        clustering = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
    elif method == "gmm":
        clustering = GaussianMixture(n_components=num_clusters, random_state=42).fit(
            features
        )
    elif method == "dbscan":
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
    else:
        raise ValueError("Unsupported clustering method")
    return clustering.labels_


labels = perform_clustering(labelled_features, method="kmeans", num_clusters=2)

# %%
class Classifier(nn.Module):
    def __init__(self, vae, device, input_dim=128, dropout=0.3):
        super(Classifier, self).__init__()
        self.vae = vae.to(device)
        self.vae.eval()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            x = x["image"].to(self.device)
            mu, _ = self.vae.encode(x)
        return self.fc(mu)

# %%
from sklearn.metrics import f1_score


def train_classifier(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        target = batch["label"].float().to(device)

        optimizer.zero_grad()
        output = model(batch).squeeze()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (output > 0.5).float()
        correct += (predicted == target).sum().item()
        total += target.size(0)

    accuracy = 100.0 * correct / total
    return total_loss / len(train_loader), accuracy


def validate_classifier(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            target = batch["label"].float().to(device)
            output = model(batch).squeeze()
            loss = criterion(output, target)

            total_loss += loss.item()
            predicted = (output > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(target.cpu().numpy())

        f1 = f1_score(np.concatenate(all_targets), np.concatenate(all_preds))
        print(f"Validation Loss: {total_loss / len(val_loader)}")
        print(f"Validation Accuracy: {100. * correct / total}")
        print(f"Validation F1 Score: {f1}")
        return total_loss / len(val_loader)

# %%
model = ConvVAE(latent_dim=256).to(device)
model.load_state_dict(
    torch.load("/data/iivanova-23/ROB313/models/vae_trial/vae_model.pth")
)
classifier = Classifier(model, device, input_dim=256)
optimizer_classifier = optim.Adam(classifier.parameters(), lr=1e-4)
criterion_classifier = nn.BCELoss()
num_epochs = 30
for epoch in range(num_epochs):
    train_loss, train_accuracy = train_classifier(
        classifier,
        train_loader_labeled,
        optimizer_classifier,
        criterion_classifier,
        device,
    )
    print(f"Epoch {epoch} Train loss: {train_loss}, Train accuracy: {train_accuracy}")

# %%
validate_classifier(classifier, val_loader, nn.BCELoss(), device)

# %%
validate_classifier(classifier, test_loader, nn.BCELoss(), device)


