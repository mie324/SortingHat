import torch.nn as nn
import torch


class NLP(nn.Module):
    def __init__(self):
        super(NLP, self).__init__()
        self.fc1 = nn.Linear(H1)
        self.fc2 = nn.Linear(H1, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x))

        return x