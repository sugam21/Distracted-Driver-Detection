# import numpy as np
# from PIL import Image
import pandas as pd
from typing import List
from typing import Optional
from typing import Callable
from torchvision.transforms import v2
from torchvision.io import read_image
import os
import torch
from torch.utils.data import Dataset


class DataTransform:
    def __init__(self, input_size: int, channel_mean, channel_std):
        self.data_transform = {
            "train": v2.Compose(
                [
                    v2.RandomResizedCrop(size=(input_size, input_size), antialias=True),
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=channel_mean, std=channel_std),
                ]
            ),
            "valid": v2.Compose(
                [
                    v2.RandomResizedCrop(size=(input_size, input_size), antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=channel_mean, std=channel_std),
                ]
            ),
        }

    def __call__(self, image, train=True):
        if train:
            transformed = self.data_transform["train"](image)
        else:
            transformed = self.data_transform["valid"](image)
        return transformed


class CustomDataset(Dataset):
    def __init__(
        self,
        annotation_files,
        img_dir: str,
        is_train: bool = True,
        transform=None,
    ):
        self.img_labels = annotation_files
        self.img_dir: str = img_dir
        self.transform = transform
        self.is_train: bool = is_train

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.img_dir,
            self.img_labels.iloc[idx, 1],
            self.img_labels.iloc[idx, -1],
        )  # gets the indexed image

        image: Tensor = read_image(img_path)  # converts the image into tensor
        label: int = self.img_labels.iloc[idx, 2]  # gets the label of the image
        if self.transform:
            image = self.transform(image, self.is_train)
        return image, label
