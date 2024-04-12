import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        # using pre-trained weights to initialize the model
        self.vgg16 = models.vgg16(weights="IMAGENET1K_V1")

    def forward(self, x):
        return self.vgg16(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = Vgg16()
    output = model.forward(x)
    print(output.shape)
