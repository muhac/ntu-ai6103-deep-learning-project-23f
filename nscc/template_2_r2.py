# -*- coding: utf-8 -*-
"""seresnet_train_template_AI6103.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RANIkJxQSc766AVZOEqluvGo40Rm3F1z
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import numpy as np
import random
import pdb
import json

"""# Setings"""

# remember to check the epoch, testing epoch is 200,
learning_rate = 0.1
weight_decay = 0.00005
epochs = 200
batch_size = 64
senet_r = 8

info = f"r_{senet_r}_lr_{learning_rate}_wd_{weight_decay}_batch_{batch_size}_epoch_{epochs}"
print(info)

acc_and_loss_saved_filname = f"experiment2_senet_{info}.json"
model_weight_filename = f"experiment2_senet_{info}.pth"

seed = 0

# Set seed for PyTorch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set seed for NumPy
np.random.seed(seed)
# Set seed for Python's random module
random.seed(seed)

"""# Models"""


class ResNet50(nn.Module):
    def __init__(self, block, num_classes=1000, r=1):
        super(ResNet50, self).__init__()
        self.block = block
        self.r = r
        self.num_classes = num_classes

        self.stage1 = self.stage_input()
        self.stage2 = self.stage_blocks(3, 64, 64, stride=1)
        self.stage3 = self.stage_blocks(4, 64 * 4, 128, stride=2)
        self.stage4 = self.stage_blocks(6, 128 * 4, 256, stride=2)
        self.stage5 = self.stage_blocks(3, 256 * 4, 512, stride=2)
        self.stage6 = self.stage_output(512 * 4, self.num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        return x

    def stage_input(self):
        conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv, bn, relu, pool)

    def stage_blocks(self, n, in_channels, out_channels, stride=1):
        layers = [self.block(in_channels, out_channels, stride=stride, r=self.r)]
        for i in range(n - 1):
            layers.append(self.block(out_channels * 4, out_channels, r=self.r))
        return nn.Sequential(*layers)

    def stage_output(self, in_channels, out_channels, stride=1):
        pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        fc = nn.Linear(in_channels, out_channels)
        return nn.Sequential(pool, flatten, fc)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU()

        self.need_downsample = stride != 1 or in_channels != out_channels * 4
        if self.need_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.need_downsample:
            identity = self.downsample(identity)

        out = x + identity
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    def __init__(self, channel, r):
        super(SEBlock, self).__init__()
        reductionChannel = channel // r
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(channel, reductionChannel)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(reductionChannel, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pdb.set_trace()
        out = self.pool(x)
        batch, channel = out.size(0), out.size(1)
        out = out.view(batch, -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.view(batch, channel, 1, 1)
        out = self.sigmoid(out)
        return out


class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r=1):
        super(SEResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU()

        self.need_downsample = stride != 1 or in_channels != out_channels * 4
        if self.need_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
        else:
            self.downsample = None

        self.seBlock = SEBlock(out_channels * 4, r)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.need_downsample:
            identity = self.downsample(identity)

        # apply SE
        scale = self.seBlock(x)

        out = x * scale.expand_as(x) + identity
        out = self.relu(out)

        return out


"""# Dataset"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

# Download CIFAR-10 training dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Download CIFAR-10 testing dataset
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train, valid = torch.utils.data.random_split(train_dataset, [40000, 10000])

# Create data loaders for training and testing datasets
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

"""# Training"""

model = ResNet50(SEResNetBlock, r=senet_r)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_loss, train_acc = [], []
valid_loss, valid_acc = [], []

for epoch in range(epochs):
    # Train loop
    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # train
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # saving for loss and acc
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_loss.append(running_loss)
    train_acc.append(correct_predictions / total_samples)
    scheduler.step()

    # Valid loop
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    running_loss = 0

    with torch.no_grad():

        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # test
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # saving for loss and acc
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    valid_loss.append(running_loss)
    valid_acc.append(correct_predictions / total_samples)

    print(
        f'Epoch {epoch + 1}/{epochs} - '
        f'Train Running Loss: {train_loss[-1]:.4f} - '
        f'Valid Loss: {valid_loss[-1]:.4f} - '
        f'Valid Accuracy: {valid_acc[-1]:.4f}'
    )

print('Training finished.')

"""# Testing"""

# Testing loop
model.eval()  # Set the model to evaluation mode
correct_predictions = 0
total_samples = 0
running_loss = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # test
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # saving for loss and acc
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

print(
    f'Epoch {epoch + 1}/{epochs} - '
    f'Train Running Loss: {train_loss[-1]:.4f} - '
    f'Test Loss: {running_loss:.4f} - '
    f'Test Accuracy: {correct_predictions / total_samples:.4f}'
)

"""# Save Data"""

torch.save(model, model_weight_filename)

saved_loss_and_acc = {
    "train_loss": train_loss,
    "train_acc": train_acc,
    "valid_loss": valid_loss,
    "valid_acc": valid_acc,
}

with open(acc_and_loss_saved_filname, 'w') as json_file:
    json.dump(saved_loss_and_acc, json_file)
