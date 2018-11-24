import torch.nn as nn
import torch

KER1 = 48
KER2 = 64
KER3 = 72

KER_SIZE1 = 7
KER_SIZE2 = 5
KER_SIZE3 = 3
PAD1 = KER_SIZE1 //2
PAD2 = KER_SIZE2 //2
PAD3 = KER_SIZE3 //2
DIL = 2
#KER_SIZE3 = 3
POOL_SIZE = 3

dim_after_conv1 = (128+2*PAD1 - DIL*(KER_SIZE1-1))//POOL_SIZE
dim_after_conv2 = (dim_after_conv1 + 2*PAD2 - DIL*(KER_SIZE2-1))//POOL_SIZE
dim_after_conv3 = (dim_after_conv2 + 2*PAD3 - 1*(KER_SIZE3-1))
print(dim_after_conv1, dim_after_conv2, dim_after_conv3)
H0 = KER3 * dim_after_conv3**2
H1 = 2048
H2 = 256
H3 = 64
HOUT = 11
P_DR = 0.5



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, KER1, KER_SIZE1, padding=PAD1, dilation=DIL)
        self.conv2 = nn.Conv2d(KER1, KER2, KER_SIZE2, padding=PAD2, dilation=DIL)
        self.conv3 = nn.Conv2d(KER2, KER3, KER_SIZE3, padding=PAD3)
        self.pool = nn.MaxPool2d(POOL_SIZE, POOL_SIZE)
        self.fc1 = nn.Linear(H0, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, H3)
        self.fc4 = nn.Linear(H3, HOUT)
        # Hin+2×padding−dilation×(kernel_size−1)
        self.bnc1 = nn.BatchNorm2d(KER1)
        self.bnc2 = nn.BatchNorm2d(KER2)
        self.bnc3 = nn.BatchNorm2d(KER3)
        self.bnf1 = nn.BatchNorm1d(H1)
        self.bnf2 = nn.BatchNorm1d(H2)
        self.bnf3 = nn.BatchNorm1d(H3)

        self.dropout = nn.Dropout(p = P_DR)

    def forward(self, x):
        x = self.pool(torch.relu(self.bnc1(self.conv1(x))))
        x = self.pool(torch.relu(self.bnc2(self.conv2(x))))
        x = torch.relu(self.bnc3(self.conv3(x)))
        x = x.view(-1, KER3 * dim_after_conv3**2)
        x = self.dropout(torch.relu(self.bnf1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bnf2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bnf3(self.fc3(x))))
        x = torch.softmax(self.fc4(x), dim=1)
        print(x.get_device())
        return x
