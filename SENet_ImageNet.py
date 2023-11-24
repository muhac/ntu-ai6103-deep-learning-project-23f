import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##########
# Read data

import os

output = './outputs/imagenet64/'
if not os.path.exists(output):
    os.makedirs(output)

datasets = './dataset/imagenet64'
checkpoints = './checkpoints/imagenet64/'
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)

os.system('ls dataset/imagenet64/')
os.system('ls checkpoints/imagenet64/')


def get_imagenet64_data():
    # Data augmentation transformations. Not for Testing!
    transform_train = transforms.Compose([
        transforms.Resize(64),  # Takes images smaller than 64 and enlarges them
        transforms.RandomCrop(64, padding=4, padding_mode='edge'),  # Take 64x64 crops from 72x72 padded images
        transforms.RandomHorizontalFlip(),  # 50% of time flip image along y-axis
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(root=datasets + '/train/', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder(root=datasets + '/val/', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    return {'train': trainloader, 'test': testloader}


data = get_imagenet64_data()

dataiter = iter(data['train'])
images, labels = next(dataiter)
images = images[:8]
print(images.size())

# show images
img = torchvision.utils.make_grid(images)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.savefig(output + 'samples-senet.png')

# print labels
print("Labels:" + ' '.join('%9s' % labels[j] for j in range(8)))

flat = torch.flatten(images, 1)
print(images.size())
print(flat.size())


##########
# ResNet50

class ResNet50(nn.Module):
    def __init__(self, block, num_classes=1000):
        super(ResNet50, self).__init__()
        self.block = block
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
        layers = [self.block(in_channels, out_channels, stride)]
        for i in range(n - 1):
            layers.append(self.block(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def stage_output(self, in_channels, out_channels, stride=1):
        pool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        fc = nn.Linear(in_channels, out_channels)
        return nn.Sequential(pool, flatten, fc)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
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
        super().__init__()
        reductionChannel = int(channel / r)
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
    def __init__(self, in_channels, out_channels, stride=1):
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

        self.seBlock = SEBlock(out_channels * 4, 1)

    def forward(self, x):
        identity = x

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.need_downsample:
            identity = self.downsample(identity)

        # apply SE
        scale = self.seBlock(x)

        out = x * scale + identity
        out = self.relu(out)

        return out


##########
# Training

def train(net, dataloader, epochs=1, start_epoch=0, lr=0.01, momentum=0.9, decay=0.0005,
          verbose=1, print_every=10, state=None, schedule={}, checkpoint_path=None):
    net.to(device)
    net.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load previous training state
    if state:
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        losses = state['losses']

    # Fast forward lr schedule through already trained epochs
    for epoch in range(start_epoch):
        if epoch in schedule:
            print("Learning rate: %f" % schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

    for epoch in range(start_epoch, epochs):
        sum_loss = 0.0

        # Update learning rate when scheduled
        if epoch in schedule:
            print("Learning rate: %f" % schedule[epoch])
            for g in optimizer.param_groups:
                g['lr'] = schedule[epoch]

        for i, batch in enumerate(dataloader, 0):
            inputs, labels = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # autograd magic, computes all the partial derivatives
            optimizer.step()  # takes a step in gradient direction

            losses.append(loss.item())
            sum_loss += loss.item()

            if i % print_every == print_every - 1:  # print every 10 mini-batches
                if verbose:
                    info = '[%d, %5d] loss: %.3f' % (epoch, i + 1, sum_loss / print_every)
                    print(info)
                    os.system('echo ' + info + ' >> ' + output + 'resnet-loss.txt')

                sum_loss = 0.0
        if checkpoint_path:
            state = {'epoch': epoch + 1, 'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'losses': losses}
            torch.save(state, checkpoint_path + 'checkpoint-senet-%d.pkl' % (epoch + 1))
    return losses


net = ResNet50(SEResNetBlock)
losses = train(net, data['train'], epochs=20, schedule={0: .1, 5: .01, 15: .001}, checkpoint_path=checkpoints)


##########
# Testing

def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode='valid')


plt.plot(smooth(losses, 50))
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(output + 'senet-loss.png')


def accuracy(net, dataloader):
    net.to(device)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


print("Training accuracy: %f" % accuracy(net, data['train']))
print("Testing  accuracy: %f" % accuracy(net, data['test']))
