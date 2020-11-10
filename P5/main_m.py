import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DataLoader, SubsetRandomSampler
from torchsummary import summary

import numpy as np
import os
import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
# from vgg16 import vgg16_f
from vgg16 import VGG_net

mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

# epoch = 20
# batch_size = 32
# learning_rate = 0.001
# optimizer = 'SGD'
# TRAIN_NUMS = 49000

def build_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epoch', type=int, help='num of training epoch', default=20)
    parser.add_argument('--batchsize', type=int, help='batchsize', default=256)
    parser.add_argument('--num-workers', type=int, help='num of workers to load data', default=8)
    parser.add_argument('--lr', type=int, help='initial learning rate', default=1e-1)
    parser.add_argument('--lr-milestone', type=list, help='list of epoch for adjust learning rate', default=[50, 150, 200])
    parser.add_argument('--lr-gamma', type=float, help='factor for decay learning rate', default=0.1)
    parser.add_argument('--momentum', type=float, help='momentum for optimizer', default=0.9)
    parser.add_argument('--weight-decay', type=float, help='factor for weight decay in optimizer', default=5e-4)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    
    return parser

class Trainer:
    def __init__(self, args,criterion, optimizer, schedular, device):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
        self.device = device

        self.train_loss_list = []
        self.val_loss_list = []
        self.training_acc = []
        self.val_acc = []
        self.testing_acc = []
    def train_loop(self, model, train_loader, val_loader, test_loader):
        for epoch in range(args.epoch):
            train_acc, train_loss = self.train_step(model, train_loader)
            val_acc, val_loss = self.validate(model, val_loader, 'val')
            test_acc = self.validate(model, test_loader,'test')
        return model

    def train_step(self, model, loader):
        model.train()
        loss_seq = []
        outputs_list = []
        target_list = []
        # progress bar
        iterator = tqdm(loader, desc='Training:', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for inputs, targets in iterator:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.criterion(outputs, targets)
            loss_seq.append(loss.item())
            loss.backward()
            self.optimizer.step()

            outputs_list.append(outputs)
            targets_list.append(targets)

        epoch_loss = sum(loss_seq) / len(loss_seq)
        self.train_loss.append(epoch_loss)

        # tensor concatnate
        targets = torch.cat(target_list)
        outputs = torch.cat(outputs_list)
        acc = self.accuracy(outputs, targets)

        self.training_acc.append(acc)

        self.schedular.step()

        torch.save({'state_dict': model.state_dict()}, './model.pth')
        return acc, epoch_loss
    
    def validate(self, model, loader, state='val'):
        model.eval()
        loss_seq = []
        outputs_list = []
        target_list = []

        with torch.no_grad():
            desc_word = 'Validating' if state == 'val' else 'Testing'
            iterator = tqdm(loader, desc=desc_word, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            for inputs, targets in iterator:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                target_list.append(targets)
                outputs_list.append(outputs)
                loss_seq.append(loss.items)
            targets = torch.cat(target_list)
            outputs = torch.cat(outputs_list)
            acc = self.accuracy(outputs,inputs)

            if state == 'val':
                epoch_loss = sum(loss_seq) / len(loss_seq)
                self.val_loss.append(epoch_loss)
                self.val_acc.append(acc)

                return acc, epoch_loss
            else:
                self.testing_acc.append(acc)
                return acc

    def accuracy(self, outputs, targets):
        predicitions = outputs.argmax(dim = 1)
        correct = float(predicitions.eq(targets).cpu().sum())
        acc = 100 * correct / target.size(0)

        return acc

        


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
    parser = build_parser()
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    path = './ CIFAR10/'
    if os.path.exists(path + 'cifar-10-batches-py'):
        download = False
    else:
        download = True
    train_dataset = torchvision.datasets.CIFAR10(root = path, train=True, transform=train_transform, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_transform, download=download)

    split = int(len(train_dataset) * 0.9)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=SubsetRandomSampler(range(split)), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=SubsetRandomSampler(range(split, len(train_dataset))), num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    network = VGG_net()
    network.cuda()

    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestone, gamma=args.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(args, criterion, optimizer, scheduler, device)
    network = trainer.train_loop(network, train_loader, val_loader, test_loader)

    train_loss = trainer.train_loss_list.copy()
    train_acc = trainer.training_acc.copy()
    val_loss = trainer.val_loss_list.copy()
    val_acc = trainer.val_acc.copy()
    test_acc = trainer.testing_acc.copy()

    x = np.arange(1, args.epoch+1, dtype=np.int)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Loss and Accuracy')

    ax1.plot(x, train_loss, x, val_loss)
    ax1.set_ylabel('Loss')
    ax1.set_xlim(1, args.epoch)
    ax1.legend(['train loss', 'val loss'])

    ax2.plot(x, train_acc, x, val_acc, x, test_acc)
    ax2.set_ylabel('Accuracy(%)')
    ax2.set_xlabel('Epoch')
    ax2.set_xlim(1, args.epoch)
    ax2.legend(['training acc', 'val acc', 'test acc'])

    plt.savefig('acc_loss.png', dpi=800)

