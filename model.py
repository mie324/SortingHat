import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self, num_kernels=(6, 16), kernel_size=(5, 5)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, num_kernels[0], kernel_size[0])
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, num_kernels[1], kernel_size[1])
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

