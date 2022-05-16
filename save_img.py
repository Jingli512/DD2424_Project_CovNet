import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import pickle
import copy

transform_train = transforms.Compose([
        transforms.ToTensor()
])
transform_test = transforms.Compose([
    transforms.ToTensor()
])
'''Loading Data'''
train_data = torchvision.datasets.CIFAR10(
        root = './data',
        train = True,
        transform = transform_train,
        download = True
)
test_data = torchvision.datasets.CIFAR10(
        root = './data',
        train = False,
        transform = transform_test,
        download = True
)
train_loader = Data.DataLoader(dataset = train_data, batch_size = 50, shuffle = False)
test_loader = Data.DataLoader(dataset = test_data, batch_size = 1, shuffle = False)

for step, (x, y) in enumerate(train_loader):
    if(step == 0):
        x_all = copy.deepcopy(x)
        y_all = copy.deepcopy(y)
    else:
        x_all = torch.cat((x_all, x), 0)
        y_all = torch.cat((y_all, y), 0)
print(x_all.shape)
print(y_all.shape)
import pickle
with open('train/train_data.pkl', 'wb') as f:
    pickle.dump(x_all, f, pickle.HIGHEST_PROTOCOL)
with open('train/train_label.pkl', 'wb') as f:
    pickle.dump(y_all, f, pickle.HIGHEST_PROTOCOL)