import math
import torch.nn as nn
import torch.nn.functional as F


class CNN_CelebA(nn.Module):
    def __init__(self, c: int):
        super(CNN_CelebA, self).__init__()

        self.c = c
        self.c_sqrt = math.sqrt(c)

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=512)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=1024)

        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(4)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, c, bias=False)  # W

    def forward(self, x):
        x = self.gn1(self.pool1(F.relu(self.conv1(x))))
        x = self.gn2(self.pool1(F.relu(self.conv2(x))))
        x = self.gn3(self.pool1(F.relu(self.conv3(x))))
        x = self.gn4(self.pool1(F.relu(self.conv4(x))))
        x = self.gn5(self.pool2(F.relu(self.conv5(x))))
        x = self.flatten(x)
        x = self.fc(x)
        x = self.c_sqrt * F.normalize(x, dim=-1)  # scaling
        return x
