import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DataLoader, SubsetRandomSampler
from torchsummary import summary

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

epoch = 5
batch_size = 64
learning_rate = 0.001
optimizer = 'SGD'
TRAIN_NUMS = 49000

def random_show_image():
    global train_loader
    pics = enumerate(train_loader_all)
    # print(pics)
    batch_idx, (data,labels) = next(pics)

    fig = plt.figure('10 Random Images')
    for i in range(10):
        index = random.randint(0, 10000)
        plt.subplot(1, 10, i+1)
        plt.tight_layout()
        plt.imshow(data[index][0],cmap='gray', interpolation='none')
        plt.title("{}".format(labels[index]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    # print(data,labels)

if __name__ == '__main__':
    # 不太懂
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
    # download dataset
    train_data = datasets.CIFAR10('./',train=True, transform=data_transform,download=True)
    # train loader
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True)

    train_loader_all = DataLoader(train_data, batch_size=10000, shuffle=True)
    print(train_loader_all)
    # test data
    test_data = datasets.CIFAR10('./',train=True, transform=data_transform,download=True)
    # test loader
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=True)
    # 不太懂為什麼要寫個all
    test_loader_all = DataLoader(test_data, batch_size=10000,shuffle=True)

    val_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(range(TRAIN_NUMS, 50000)))
    random_show_image()

