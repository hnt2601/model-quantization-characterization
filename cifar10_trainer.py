from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision import datasets
from torch import optim
from tqdm import tqdm
from backbone import *
from utils import EarlyStopper

import torch.nn as nn
import numpy as np
import torch
import timm
import os


device = "cuda:0" if torch.cuda.is_available() else "cpu"

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

data_path = "/media/data/hoangnt/Projects/Datasets"

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

batch_size = 256

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


trainset = datasets.CIFAR10(
    root=data_path, train=False, download=False, transform=transform_train
)
train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
)

testset = datasets.CIFAR10(
    root=data_path, train=False, download=False, transform=transform_test
)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
)


# Define a Network
# model = MobileNetV2(class_num=len(classes), pretrained=True)
# model = EfficientnetV2(class_num=len(classes), pretrained=True)
model = ResNet50(class_num=len(classes), pretrained=True)
model.to(device)

# Training Hyperparameters
lr = 1
optimizer = optim.Adadelta(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# Loss Function
loss_function = nn.CrossEntropyLoss()


def train_step(epoch, loss_train_epoch):
    model.train()
    epoch_loss = []
    loop = tqdm(enumerate(train_loader))
    loop.set_description(f"Epoch {epoch}")
    for batch_idx, (data, target) in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        loop.set_postfix(
            {
                "Iter": "[{}/{}]".format(
                    batch_idx * len(data), len(train_loader.dataset)
                ),
                "Epoch_Loss": sum(epoch_loss) / len(epoch_loss),
                "Batch_Loss": loss.item(),
            }
        )
    loss_train_epoch.append(sum(epoch_loss) / len(epoch_loss))

    return loss_train_epoch


def test(best_auc, loss_test_epoch, epoch_auc_history):
    model.eval()
    test_loss_list = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss_list.append(loss_function(output, target).item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss = sum(test_loss_list) / len(test_loss_list)
    loss_test_epoch.append(test_loss)
    epoch_auc_history.append(correct / 100)

    if correct > best_auc:
        best_auc = correct
        torch.save(model.state_dict(), f"pretrained/{model.name}.pth")

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    print("Best Accuracy :", best_auc)

    return best_auc, loss_test_epoch, epoch_auc_history


# Training Section

epoches = 200
loss_train_epoch = []
epoch_auc_history = []
loss_test_epoch = []
best_auc = 0

early_stopper = EarlyStopper(patience=3, min_delta=10)

for epoch in range(1, epoches):
    loss_train_epoch = train_step(epoch, loss_train_epoch)
    best_auc, loss_test_epoch, epoch_auc_history = test(
        best_auc, loss_test_epoch, epoch_auc_history
    )

    if early_stopper.early_stop(loss_test_epoch[-1]):
        break

    scheduler.step()
