import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, out_features, pretrained=True, train_backbone=False):
        super().__init__()
        if pretrained:
            self.resnet = models.resnet50(weights="DEFAULT")
        else:
            self.resnet = models.resnet50(weights=None)
        if not train_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        in_features = self.resnet.fc.in_features
        print(f"Using ResNet50 with {in_features} FC input features")
        self.resnet.fc = ProjectionHead(in_features, out_features)

    def forward(self, x):
        return self.resnet(x)
