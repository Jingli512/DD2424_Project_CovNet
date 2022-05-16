import torch
import torchvision
from torchvision import transforms
import os
from PIL import Image

class Custom_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, imgs, labels, transform = None):
        self.imgs = imgs
        self.labels = labels
        print(f'Customized CIFAR-10, data: {self.imgs.shape}, labels: {self.labels.shape}')
        self.transform = transform
 
    def __getitem__(self, index):
        img = self.imgs[index, ...]
        label = self.labels[index]

        if self.transform is not None:
            img = transforms.ToPILImage()(img).convert('RGB')
            img = self.transform(img)
        return img, label
 
    def __len__(self):
        return self.imgs.shape[0]