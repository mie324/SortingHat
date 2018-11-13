import torch.nn as nn
import torch

KER1 = 15
KER2 = 20
#KER3 = KER2
KER_SIZE1 = 5
KER_SIZE2 = 5
PAD1 = KER_SIZE1 //2
PAD2 = KER_SIZE2 //2
#KER_SIZE3 = 3
POOL_SIZE = 4
H0 = 128 // (POOL_SIZE**2) #1280
H1 = 200
H2 = 64
HOUT = 11
P_DR = 0.5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, KER1, KER_SIZE1, padding=PAD1)
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.conv2 = nn.Conv2d(KER1, KER2, KER_SIZE2, padding=PAD2)
        self.fc1 = nn.Linear(KER2 * H0 * H0, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, HOUT)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, KER2 * H0 * H0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x

