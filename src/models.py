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
    def __init__(self, in_features, out_features, use_bn = True, hidden_layers=[]):
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
        print(f"Using ResNet50 with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(in_features, out_features, use_bn=use_bn, hidden_layers=hidden_layers)

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
