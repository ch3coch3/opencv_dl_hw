import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from torch.utils.data import Dataloader
import torch.nn as nn
import torchvision.models as models
import torchsummary as summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

train_loss_list = []
train_acc = []
test_acc = []

def train_loop(model, train_loader, test_loader):
    for epoch in range(epochs):
        train_acc, train_loss = train_step(model,train_loader)
        test_acc = validate(model, test_loader)

    return model
        

def train_step(model,loader):
    model.train()
    loss_seq = []
    output_list = []
    label_list = []
    loader = tqdm(loader, desc='Training:', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    for inputs, labels in loader:
        # data to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # forward
        outputs = model(inputs)
        # loss
        loss = criterion(outputs, labels)
        loss_seq.append(loss.item())
        # backward
        loss.backward()
        # optimizer updates paremeters
        optimizer.step()
        
        output_list.append(outputs)
        label_list.append(labels)
    # calculate accuracy
    epoch_loss = sum(loss_seq) / len(loss_seq)
    train_loss_list.append(epoch_loss)
    
    labels = torch.cat(label_list)
    outputs = torch.cat(output_list)
    acc = accuraccy(outputs, labels)
    train_acc.append(acc)

    scheduler.step()
    torch.save({'state_dict': model.state_dict()}, './model.pth')
    return acc, epoch_loss

def validate(model,loader):
    model.eval()
    loss_seq = []
    output_list = []
    label_list = []
    with torch.no_grad():
        loader = tqdm(loader, desc=desc_word, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for inputs, labels in loader:
            # data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, labels)
            loss_seq.append(loss.item())
            
            output_list.append(outputs)
            label_list.append(labels)
        
        labels = torch.cat(label_list)
        outputs = torch.cat(output_list)
        acc = accuraccy(outputs, labels)
        train_acc.append(acc)
        return acc



def accuraccy(self, outputs, targets):
    predictions = outputs.argmax(dim=1)
    correct = float(predictions.eq(targets).cpu().sum())
    acc = 100 * correct / targets.size(0)

    return acc 

if __name__ == "__main__":
    
    # transform
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)
                                    ])

    # training
    batch_size = 32
    learning_rate = 0.001
    optimizer = 'SGD'
    epochs = 20
    if torch.cuda.is_available():
        key = "cuda" 
    else:
        key = "cpu"
    device = torch.device(key)

    # data
    train_data = torchvision.datasets.CIFAR10('./CIFAR10/',
                                            train=True,
                                            transform=transform,
                                            download=True)
    test_data = torchvision.datasets.CIFAR10('./CIFAR10/',
                                            train=False,
                                            transform=transform,
                                            download=True)

    # dataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=4)
    test_loader = DataLoader(train_data, batch_size=batch_size,shuffle=True,num_workers=4)


    # VGG16 Model
    VGG16 = models.vgg16()

    VGG16.cuda()

    # set loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(VGG16.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

    # training
    train_loop(VGG16,train_loader,test_loader)


    x = np.arange(1, epochs+1, dtype=np.int)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Loss and Accuracy')

    ax1.plot(x, train_acc, x, test_acc)
    ax1.set_ylabel('Accuracy(%)')
    ax1.set_xlim(1, epochs)
    ax1.legend(['training', 'testing'])

    ax2.plot(x, train_loss_list)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_xlim(1, epochs)
    # ax2.legend(['training acc', 'val acc', 'test acc'])

    plt.savefig('acc_loss.png', dpi=800)


