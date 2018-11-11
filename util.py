import torchvision 
import torchvision.transforms as transforms
import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

CLASSES = {'plasticbottle':0, 'newspaper':1, 'plasticbag':2, 'perishable':3,
            'glassbottle':4, 'popcan':5, 'juicebox':6, 'ffwrapper':7,
            'snackpackage':8, 'coffeecups':9, 'togobox':10}
CLASSES_itos = ['plasticbottle', 'newspaper', 'plasticbag', 'perishable',
            'glassbottle', 'popcan', 'juicebox', 'ffwrapper',
            'snackpackage', 'coffeecups', 'togobox']

class WasteDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_train_val_loader(bs=64):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # pbot: 9696
    # news: 12416
    # pbag: 8532
    # peri: 8584
    train_tensors, val_tensors = [], []
    train_labels, val_labels = [], []

    for clsname, clscode in CLASSES.items():
        print('--- at {} ---'.format(clsname))
        numpics = len(os.listdir('./data/{}'.format(clsname)))
        #print(clsname, numpictures)
        rnd_sample = np.random.choice(np.arange(1, numpics+1), 8000, replace=False)

        for i,fnum in enumerate(rnd_sample):  #0 -- 7999
            filename = './data/{}/{}{}.png'.format(clsname, clsname, fnum)
            im = Image.open(filename)
            imtensor = transform(im)
            if i < 6400:
                train_tensors.append(imtensor)
                train_labels.append(clscode)
            else:
                val_tensors.append(imtensor)
                val_labels.append(clscode)
        print(len(train_labels), len(val_labels))


    X_train = torch.stack(train_tensors)
    X_val = torch.stack(val_tensors)
    y_train = torch.Tensor(train_labels)
    y_val = torch.Tensor(val_labels)
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    train_loader = DataLoader(WasteDataset(X_train, y_train), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WasteDataset(X_val, y_val), batch_size=bs, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    get_train_val_loader()


