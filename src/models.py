import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Union

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
    def __init__(self, in_features, out_features, use_bn=True, hidden_layers=[]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm1d(in_features)

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        out_features,
        backbone="resnet50",
        pretrained=True,
        train_backbone=False,
        hidden_layers=[],
        use_bn=True,
    ):
        super().__init__()
        if pretrained:
            self.resnet = BACKBONES[backbone](weights="DEFAULT")
        else:
            self.resnet = BACKBONES[backbone](weights=None)
        if not train_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        print(f"Using {backbone} with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(
            in_features, out_features, use_bn=use_bn, hidden_layers=hidden_layers
        )

    def forward(self, x):
        return self.resnet(x)


class ResNetBinaryClassifier(nn.Module):
    def __init__(
        self, backbone="resnet50", pretrained=True, train_backbone=False, path=None
    ):
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
        print(f"Using {backbone} with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(in_features, 1)
        self.resnet.fc.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.resnet(x))


class ResNetCoordsBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone="resnet50",
        pretrained=True,
        train_backbone=False,
        hidden_dims=[512],
        dropout=0.3,
        path=None,
    ):
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
        print(f"Using {backbone} with {in_features} FC input features")
        self.resnet.fc = nn.Identity()

        self.mlp = nn.Sequential()
        current_in_features = in_features + 2
        for i, hidden_dim in enumerate(hidden_dims):
            self.mlp.add_module(f"fc{i}", nn.Linear(current_in_features, hidden_dim))
            if i == 0:
                self.mlp.add_module(f"bn{i}", nn.BatchNorm1d(hidden_dim))
            self.mlp.add_module(f"relu{i}", nn.ReLU())
            self.mlp.add_module(f"dropout{i}", nn.Dropout(dropout))
            current_in_features = hidden_dim
        self.mlp.add_module("fc_out", nn.Linear(current_in_features, 1))
        self.mlp.add_module("sigmoid", nn.Sigmoid())

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x, coords):
        x = self.resnet(x)
        x = torch.cat([x, coords], dim=1)
        return self.mlp(x)


class BinaryClassifierWithPretrainedEncoder(nn.Module):
    def __init__(self, encoder, tune_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.tune_encoder = tune_encoder
        in_features = encoder.resnet.fc.out_features
        self.classifier = ProjectionHead(in_features, 1)
        self.sigmoid = nn.Sigmoid()
        if not tune_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        if not self.tune_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        return self.sigmoid(self.classifier(x))


    

class ConvVAE(nn.Module):
    def __init__(self, 
                 latent_dim=256,
                 pretrained=True, 
                 backbone="resnet50", 
                 beta=1.0):
        super(ConvVAE, self).__init__()
        
        self.pretrained = pretrained
        self.latent_dim = latent_dim
        self.beta = beta # for KL divergence
        
        if self.pretrained:
            
            resnet = BACKBONES[backbone](weights="DEFAULT")
            modules = list(resnet.children())[:-1]  
            self.resnet = nn.Sequential(*modules)
            self.encoder_output_dim = resnet.fc.in_features 
            
        else:
            # Standard convolutional encoder
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
                nn.ReLU()
            )
            self.encoder_output_dim = 256 * 7 * 7  
        
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_var = nn.Linear(self.encoder_output_dim, latent_dim)
        
        # Decoder input       
        self.decoder_input = nn.Linear(latent_dim, 256 * 7 * 7)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 28 -> 56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 112 -> 224
            nn.Sigmoid()
        )

    def encode(self, x):
        batch_size = x.size(0)
        if self.pretrained:
            x = self.resnet(x) 
            x = x.view(batch_size, -1)  
        else:
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
        x_reconstructed = self.decode(z)
        return [x, x_reconstructed, mu, log_var]
    
    def loss_function(self, *args):
        x = args[0]
        recon_x = args[1]
        mu = args[2]
        log_var = args[3]   
        
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - (log_var.exp() + 1e-8))
        loss = recon_loss + self.beta * kl_loss        
        return {'loss': loss}
    
    
    def __repr__(self):
        return "ConvVAE"
    
class ClassifierFeatures(nn.Module):
    def __init__(self, vae, input_dim=256, dropout=0.1, coords=False):
        super(ClassifierFeatures, self).__init__()
        self.vae = vae
        input_dim = input_dim + 2 if coords else input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, coords = None):
        with torch.no_grad():
            if self.vae.__class__.__name__ == "VQVAE":
                x = self.vae.encode(x)
                x, _ = self.vae.vq_layer(x)
                x = x.view(x.shape[0], -1) 
            else:
                mu, logvar = self.vae.encode(x)
                x = self.vae.reparameterize(mu, logvar)
        if coords is not None:
            x = torch.cat([x, coords], dim=1)
        return self.fc(x)
    
    def train(self):
        self.fc.train()
        self.vae.eval()
        
    def eval(self):
        self.fc.eval()
        self.vae.eval()
    
class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)

class VQVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 embedding_dim: int = 64,
                 num_embeddings: int = 512,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 pretrained=True,
                 backbone="resnet50",
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        
        self.pretrained = pretrained
        # Resnet Encoder
        if self.pretrained:
            resnet = BACKBONES[backbone](weights="DEFAULT")
            modules = list(resnet.children())[:-1]
            self.resnet = nn.Sequential(*modules)
            in_channels = resnet.fc.in_features

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        # Build Encoder
        kernel_size = 3 if self.pretrained else 4
        stride = 1 if self.pretrained else 2
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel_size, stride=stride, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )
        if self. pretrained:
            modules.append(
            nn.Sequential( 
                          nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1] // 2, kernel_size=4, stride=2, padding=1),
                          nn.LeakyReLU(),
                          nn.ConvTranspose2d(hidden_dims[-1] // 2, out_channels=3, kernel_size=4, stride=2, padding=1),
                          nn.Tanh()))
            modules.append(nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False))
        else:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                       out_channels=3,
                                       kernel_size=4,
                                       stride=2, padding=1),
                    nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if self.pretrained:
            input = self.resnet(input)
        result = self.encoder(input)
        return result

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        z = self.decode(quantized_inputs)
        return [z, input , vq_loss, quantized_inputs]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> torch.Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    
    def __repr__(self):
        return "VQVAE"
        
    