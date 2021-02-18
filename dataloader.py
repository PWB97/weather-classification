import torch
import numpy as np
import glob
import cv2
import PIL

from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from torchvision import transforms as T

import utils
from cutout import Cutout
from auto_augment import AutoAugment, Randaugment

class WeatherDataset(Dataset):
    def __init__(self, images, labels, transforms, output_name=False):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.output_name = output_name

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = utils.load_image(self.images[idx])
        # image = self.images[idx]
        label = utils.to_tensor(self.labels[idx], torch.long)

        if self.transforms is not None:
            image = self.transforms(image)

        if self.output_name:
            return image, label, self.images[idx]

        return image, label

class TestDataset(Dataset):
    def __init__(self, images, names, transforms):
        self.images = images
        self.names = names
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # image = utils.load_image(self.images[idx])
        image = self.images[idx]
        name = self.names[idx]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, name

class CamDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = utils.load_image(self.images[idx])
        image = self.images[idx]
        label = utils.to_tensor(self.labels[idx], torch.long)

        if self.transforms is not None:
            t_image = self.transforms(image)
            image = resize_transform(image)

        return t_image, label, np.array(image)


class UnNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def my_transform(train=True, resize=224, use_cutout=False, n_holes=1, length=8, auto_aug=False
                 , raug=False, N=0, M=0
                 ):
    transforms = []

    if train:
        transforms.append(T.RandomRotation(90))
        transforms.append(T.RandomResizedCrop(resize+20,
                          scale=(0.2, 1.0),
                          interpolation=PIL.Image.BICUBIC))
        transforms.append(T.RandomHorizontalFlip())
        # transforms.append(T.RandomVerticalFlip())
        transforms.append(T.ColorJitter(0.3, 0.2, 0.2, 0.2))
        transforms.append(T.CenterCrop(resize))
        if auto_aug:
            transforms.append(AutoAugment())
        if raug:
            transforms.append(Randaugment(N, M))

    else:
        transforms.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
        transforms.append(T.CenterCrop(resize))

    transforms.append(T.ToTensor())
    transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if train and use_cutout:
        transforms.append(Cutout())

    return T.Compose(transforms)

def test_transform(resize=224):
    transforms = []
    transforms.append(T.Resize(resize, interpolation=PIL.Image.BICUBIC))
    transforms.append(T.CenterCrop(resize))
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

def resize_transform(images, resize=224):
    transforms = []

    transforms.append(T.Resize(resize+20))
    transforms.append(T.CenterCrop(resize))

    # transforms.append(T.ToTensor())

    return T.Compose(transforms)(images)





