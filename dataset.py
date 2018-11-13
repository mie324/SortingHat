import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        sample = self.x[index]
        label = self.y[index]
        return sample, label
