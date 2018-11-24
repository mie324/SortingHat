import torch.utils.data as data


class Dataset(data.Dataset):
    '''
    deprecated. implemented in util.py
    '''
    def __init__(self, IDs, numpics):
        self.IDs = IDs  # a dictionary {classname: [ids]... } where ids contains only good pictures
        self.numpics = numpics

    def __len__(self):
        return len(self.IDs)*self.numpics

    def __getitem__(self, index):
        y, localindex = index // self.numpics, index % self.numpics

        picnum = self.IDs[y][localindex]

        return X, y

