import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

BACKBONES = {
    "resnet50": models.resnet50,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet101": models.resnet101,
}

class JustCoords(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, out_features)

        self.bn1 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(self, out_features, backbone = "resnet50", pretrained=True, train_backbone=False):
        super().__init__()
        if pretrained:
            self.resnet = BACKBONES[backbone](weights="DEFAULT")
        else:
            self.resnet = BACKBONES[backbone](weights=None)
        if not train_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        print(f"Using ResNet50 with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(in_features, out_features)

    def forward(self, x):
        return self.resnet(x)

class ResNetBinaryClassifier(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, train_backbone=False, path=None):
        super().__init__()
        if pretrained:
            if path:
                self.resnet = BACKBONES[backbone](weights="DEFAULT", path=path)
            else:
                self.resnet = BACKBONES[backbone](weights="DEFAULT")
        else:
            self.resnet = BACKBONES[backbone](weights=None)
        if not train_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        print(f"Using ResNet50 with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(in_features, 1)
        self.resnet.fc.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.resnet(x))
    

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvVAE, self).__init__()
        
        # Encoder
        # 3x224x224
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 112 -> 56
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 56 -> 28
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            # nn.Conv2d(256, 256, 4, stride=2, padding=1),  # 14 -> 7
            # nn.ReLU()
        )
        
        self.encoder_output_dim = (128 * 14 * 14)
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Linear(128, self.encoder_output_dim)
        ) 
        
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size,-1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 128, 14, 14) 
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = self.decode(z)
        return z, mu, log_var
    
class ClassifierFeatures(nn.Module):
    def __init__(self, vae, device, input_dim=128, dropout=0.3):
        super(ClassifierFeatures, self).__init__()
        self.vae = vae.to(device)  
        self.vae.eval()  
        self.device = device
        self.fc = nn.Sequential(
            # nn.Linear(input_dim, 256),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(device)  

    def forward(self, x):
        with torch.no_grad():
            x = x['image'].to(self.device)  
            mu, _ = self.vae.encode(x)  
        return self.fc(mu)