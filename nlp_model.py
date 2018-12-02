import torch.nn as nn
import torch

H1 = 32


class NLP(nn.Module):
    def __init__(self):
        super(NLP, self).__init__()
        self.fc1 = nn.Linear(300, H1)
        self.fc2 = nn.Linear(H1, 4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=1)

        return x
