import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(weights="DEFAULT")
        self.in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(
            in_features=self.in_features, out_features=10, bias=True
        )

    def get_model(self):
        return self.resnet

    def forward(self, x):
        return self.resnet(x)

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    resnet = Resnet18().get_model()
    print(resnet(x))
