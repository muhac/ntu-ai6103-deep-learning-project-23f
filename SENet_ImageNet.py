import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random

seed = 42

# Set seed for PyTorch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set seed for NumPy
np.random.seed(seed)

# Set seed for Python's random module
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

##########
# Read data


output = './outputs/imagenet64/'
if not os.path.exists(output):
    os.makedirs(output)

datasets = './dataset/imagenet64'
checkpoints = './checkpoints/imagenet64/'
if not os.path.exists(checkpoints):
    os.makedirs(checkpoints)

os.system('ls dataset/imagenet64/')
os.system('ls checkpoints/imagenet64/')


def get_dataset():
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.RandomCrop(64, padding=4, padding_mode='edge'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    data_train = torchvision.datasets.ImageFolder(root=datasets + '/train/', transform=transform_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=512, shuffle=True, num_workers=2)

    data_test = torchvision.datasets.ImageFolder(root=datasets + '/val/', transform=transform_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=512, shuffle=False, num_workers=2)

    return {'train': loader_train, 'test': loader_test}


data = get_dataset()

images, labels = next(iter(data['train']))
images = images[:8]
labels = labels[:8]
print(images.size())

img = torchvision.utils.make_grid(images)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.savefig(output + 'samples-senet.png')
print("Labels:" + ' '.join(map(str, labels)))


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


##########
# SENet Blocks

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

def step_learning_rate(schedule, default=0.01):
    def get_learning_rate(epoch):
        lr = default
        for i in range(epoch + 1):
            if i in schedule:
                lr = schedule[i]
        return lr

    return get_learning_rate


def train(model, dataloader, total_epochs=1, start_epoch=0,
          lr=0.01, momentum=0.9, decay=0.0005, schedule_func=None,
          checkpoint_model=None, checkpoint_path=None):
    model.to(device)
    model.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

    # Load checkpoint model
    if checkpoint_model:
        model.load_state_dict(checkpoint_model['net'])
        optimizer.load_state_dict(checkpoint_model['optimizer'])
        start_epoch = checkpoint_model['epoch']
        losses = checkpoint_model['losses']

    for epoch in range(start_epoch, total_epochs):
        # Update learning rate
        if schedule_func is not None:
            lr = schedule_func(epoch)
        for params in optimizer.param_groups:
            params['lr'] = lr

        for batch, batch_data in enumerate(dataloader, 0):
            xs = batch_data[0].to(device)
            ys = batch_data[1].to(device)

            optimizer.zero_grad()

            outputs = model(xs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            info = f'"Epoch {epoch}, Batch {batch}: loss: {loss.item():.3f} (LR={lr})"'
            os.system('echo ' + info + ' >> ' + output + 'senet-loss.txt')

        # Save checkpoint every 2 epochs
        if epoch % 2 == 1 and checkpoint_path:
            checkpoint_model = {
                'epoch': epoch + 1,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses,
            }
            torch.save(checkpoint_model, checkpoint_path + 'checkpoint-senet-%d.pkl' % (epoch + 1))
    return losses


net = ResNet50(SEResNetBlock)
train_losses = train(
    net, data['train'], 20,
    schedule_func=step_learning_rate({0: .1, 5: .01, 15: .001}),
    checkpoint_path=checkpoints,
)


##########
# Testing

def smooth(x, size):
    return np.convolve(x, np.ones(size) / size, mode='valid')


plt.plot(smooth(train_losses, 50))
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig(output + 'senet-loss.png')


def accuracy(model, dataloader):
    model.to(device)
    model.eval()
    corrects = 0
    total = 0
    with torch.no_grad():
        for batch_data in dataloader:
            xs = batch_data[0].to(device)
            ys = batch_data[1].to(device)
            outputs = model(xs)
            _, ys_hat = torch.max(outputs.data, 1)
            total += ys.size(0)
            corrects += (ys == ys_hat).sum().item()
    return corrects / total


print("Training accuracy: %f" % accuracy(net, data['train']))
print("Testing  accuracy: %f" % accuracy(net, data['test']))
