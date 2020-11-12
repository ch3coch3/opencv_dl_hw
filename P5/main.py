import cv2
from vgg import vgg16
import random
import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from train import Trainer

mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
path = './CIFAR10/'
def showTrainImage():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    label = ['airplane','automobile','bird','cat','deer',
            'dog','frog','horse','ship','truck']
    path = './CIFAR10/'
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])    

    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    pics = enumerate(train_loader)
    batch_idx, (data, labels) = next(pics)
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

def showParameter():
    print("hyperparameters:")
    print("batch size:",32)
    print("learning rate:", 0.001)
    print("optimizer:","SGD")

def showmodel():
    network = vgg16(num_classes=10)
    network = network.cuda()
    summary(network, (3, 224, 224))
    # print(network)

def showAcc():
    img = cv2.imread('acc_loss.png')
    dim = img.shape
    img = cv2.resize(img, (int(dim[1]/5),int(dim[0]/5)))
    # cv2.resizeWindow(img,60,60)
    cv2.imshow("loss image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])    

    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    model = torch.load('model.pth')
    acc = Trainer.validate(model, test_loader[1], 'test')

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])    

train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, transform=train_transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=256,  shuffle=False, pin_memory=True)
pics = enumerate(train_loader)
batch_idx, (data, labels) = next(pics)

data = np.squeeze(data)
print(data)
model = torch.load('model.pth')
model.eval()
acc = Trainer.validate(model, test_loader, 'test')