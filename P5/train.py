import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np

import os
import random
import argparse
from tqdm import tqdm

from vgg import vgg16

mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

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
    def __init__(self, args, criterion, optimizer, scheduler, device):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_loss_list = []
        self.val_loss_list = []
        self.training_acc = []
        self.val_acc = []
        self.testing_acc = []
    
    def train_loop(self, model, train_loader, val_loader, test_loader):
        for epoch in range(args.epoch):
            print('---------- Epoch {} ------------'.format(epoch+1))            
            train_acc, train_loss = self.train_step(model, train_loader)
            val_acc, val_loss = self.validate(model, val_loader, 'val')
            test_acc = self.validate(model, test_loader, 'test')

            print('Training loss: {:.3f}, training acc: {:.3f}; Val loss: {:.3f}, Val acc: {:.3f}; Testing acc: {:.3f}'.format(train_loss, train_acc, val_loss, val_acc, test_acc))            

        return model

    def train_step(self, model, loader):
        model.train()
        loss_seq = []
        outputs_list = []
        targets_list = []
        
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
        self.train_loss_list.append(epoch_loss)
        
        targets = torch.cat(targets_list)
        outputs = torch.cat(outputs_list)
        acc = self.accuraccy(outputs, targets)        
        self.training_acc.append(acc)
    
        self.scheduler.step()

        torch.save({'state_dict': model.state_dict()}, './model.pth')
        return acc, epoch_loss
    
    def validate(self, model, loader, state='val'):
        model.eval()
        loss_seq = []
        outputs_list = []
        targets_list = []
        
        with torch.no_grad():
            desc_word = 'Validating' if state == 'val' else 'Testing'
            iterator = tqdm(loader, desc=desc_word, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for inputs, targets in iterator:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = self.criterion(outputs, targets)

                targets_list.append(targets)
                outputs_list.append(outputs)
                loss_seq.append(loss.item())
            
            targets = torch.cat(targets_list)
            outputs = torch.cat(outputs_list)
            acc = self.accuraccy(outputs, targets)
            
            if state == 'val':
                epoch_loss = sum(loss_seq) / len(loss_seq)
                self.val_loss_list.append(epoch_loss)
                self.val_acc.append(acc)

                return acc, epoch_loss
            else:
                self.testing_acc.append(acc)
        
                return acc
    
    def accuraccy(self, outputs, targets):
        predictions = outputs.argmax(dim=1)
        correct = float(predictions.eq(targets).cpu().sum())
        acc = 100 * correct / targets.size(0)

        return acc



if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])    
    
    path = './CIFAR10/'
    train_dataset = torchvision.datasets.CIFAR10(root=path, train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_transform, download=True)

    split = int(len(train_dataset) * 0.9)
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=SubsetRandomSampler(range(split)), num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=SubsetRandomSampler(range(split, len(train_dataset))), num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    network = vgg16(num_classes=10)
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
