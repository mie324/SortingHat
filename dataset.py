import torch.utils.data as data


class Dataset(data.Dataset):
'''
Trying to stream data from disk instead of storing everything in memory
'''
    def __init__(self, IDs, labels):
        self.labels = labels
        self.list_IDs = IDs  # a dictionary {classname: [ids]... } where ids contains only good pictures

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        return sample, label
