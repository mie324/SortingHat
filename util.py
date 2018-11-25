import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import itertools
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from re import sub
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

CLASSES = {'plasticbottle':0, 'newspaper':1, 'plasticbag':2, 'perishable':3,
            'glassbottle':4, 'popcan':5, 'juicebox':6, 'ffwrapper':7,
            'snackpackage':8, 'coffeecups':9, 'togobox':10}
CLASSES4 = {'plasticbottle':1, 'newspaper':2, 'plasticbag':0, 'perishable':0,
            'glassbottle':1, 'popcan':1, 'juicebox':1, 'ffwrapper':0,
            'snackpackage':0, 'coffeecups':3, 'togobox':1, 'paper':2}
CLASSES_itos = ['plasticbottle', 'newspaper', 'plasticbag', 'perishable',
            'glassbottle', 'popcan', 'juicebox', 'ffwrapper',
            'snackpackage', 'coffeecups', 'togobox']
CLASSES4_itos = ['landfill', 'container', 'paper', 'coffeecups']

class WasteDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class WasteDatasetStream(Dataset):
    '''
    Trying to stream data from disk instead of storing everything in memory
    '''

    def __init__(self, IDs, numpics, fourclass=False, transfer=False):
        self.IDs = IDs  # a dictionary {classname: [ids]... } where ids contains only good pictures
        self.numpics = numpics  # per class
        self.classes_itos = CLASSES4_itos if fourclass else CLASSES_itos
        self.fourclass = fourclass
        if not transfer:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(299), # 299 for inception / 224 for resnet
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        ncls = 4 if self.fourclass else 11
        return ncls * self.numpics

    def __getitem__(self, index):
        clscode, localindex = index // self.numpics, index % self.numpics

        picnum = self.IDs[clscode][localindex]
        if not self.fourclass:
            clsname = self.classes_itos[clscode]
            filename = './data/{}/{}{}.png'.format(clsname, clsname, picnum)
        else:
            clsname = sub(r"[^A-Za-z]+", '', picnum)  # keep alphabetic portion of string.
            # (in fourclass, picnum is a string that include the class name)
            filename = './data/{}/{}.png'.format(clsname, picnum)
        im = Image.open(filename)
        imtensor = self.transform(im)
        try:
            assert imtensor.shape == (3,128,128)
        except AssertionError:
            print('ASSERTION EROR, ', imtensor.shape)

        return imtensor, clscode


def stream_train_val_loader(bs=64, debug=False, fourclass=False, transfer=False):

    if not fourclass:
        num_data = 1000 if debug else 8000
    else:
        num_data = 2000 if debug else 16000
    print('num_data = {}'.format(num_data))
    classes = CLASSES4 if fourclass else CLASSES

    clscode_to_numsample = [num_data // 4, num_data // 5, num_data // 2, num_data]  # used for fourclass

    train_IDs = {}
    val_IDs = {}

    for clsname, clscode in classes.items():
        numsample = clscode_to_numsample[clscode] if fourclass else num_data
        goodindices = []
        for fname in os.listdir('./data/{}'.format(clsname)):
            if '_bad' not in fname:
                if fourclass:
                    goodindices.append( os.path.splitext(fname)[0] ) # extract index
                else:
                    goodindices.append( os.path.splitext(fname)[0].split(clsname)[1] )

        rnd_sample = np.random.permutation(goodindices)

        if fourclass and clscode in train_IDs:
            train_IDs[clscode] = np.concatenate( (train_IDs[clscode], rnd_sample[:int(numsample*0.8)]) )
            val_IDs[clscode] = np.concatenate( (val_IDs[clscode], rnd_sample[int(numsample*0.8):numsample] ) )
        else:
            train_IDs[clscode], val_IDs[clscode] = rnd_sample[:int(numsample * 0.8)], rnd_sample[int(numsample * 0.8):numsample]

    print(len(train_IDs[0]), len(train_IDs[1]), len(train_IDs[2]), len(train_IDs[3]))
    train_loader = DataLoader(WasteDatasetStream(train_IDs, int(num_data*0.8), fourclass=fourclass, transfer=transfer),
                              batch_size=bs, shuffle=True, num_workers=1)
    val_loader = DataLoader(WasteDatasetStream(val_IDs, int(num_data*0.2), fourclass=fourclass, transfer=transfer),
                            batch_size=bs, shuffle=False, num_workers=1)
    logging.info('loaders generated')
    return train_loader, val_loader



def get_train_val_loader(bs=64, debug=False, fourclass=False):
    '''
    kills memory
    '''
    if not fourclass:
        num_data = 1000 if debug else 8000
    else:
        num_data = 2000 if debug else 16000

    logging.info('fourclass is {}, debug is {}'.format(fourclass, debug))
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_tensors, val_tensors = [], []
    train_labels, val_labels = [], []

    classes = CLASSES4 if fourclass else CLASSES
    clscode_to_numsample = [num_data//4, num_data//5, num_data//2, num_data]
    for clsname, clscode in classes.items():
        numsample = clscode_to_numsample[clscode] if fourclass else num_data
        #print('--- at {} ---'.format(clsname))
        numpics = len(os.listdir('./data/{}'.format(clsname)))
        #print(clsname, numpictures)
        rnd_sample = np.random.permutation(np.arange(1, numpics+1))

        success=0
        for fnum in rnd_sample:  #0 -- 7999

            filename = './data/{}/{}{}.png'.format(clsname, clsname, fnum)
            try:
                im = Image.open(filename)
            except FileNotFoundError:
                continue
            imtensor = transform(im)
            # if imtensor.shape != (3,128,128):
            #     continue

            if success < numsample*0.8:
                train_tensors.append(imtensor)
                train_labels.append(clscode)
            elif success < numsample:
                val_tensors.append(imtensor)
                val_labels.append(clscode)
            else:
                break
            success+=1
        #print(train_tensors[-1].shape, val_tensors[-1].shape)

    # for i,tensor in enumerate(train_tensors):
    #     if tensor.shape != (3, 128, 128):
    #         print('{}th tensor has dimension {}'.format(i, tensor.shape))

    X_train = torch.stack(train_tensors)
    X_val = torch.stack(val_tensors)
    y_train = torch.Tensor(train_labels)
    y_val = torch.Tensor(val_labels)
    print(X_train.shape, X_val.shape)
    print(y_train.shape, y_val.shape)

    train_loader = DataLoader(WasteDataset(X_train, y_train), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WasteDataset(X_val, y_val), batch_size=bs, shuffle=False)
    logging.info('loaders geenrated')
    return train_loader, val_loader


def examine_pic_dimensions():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    correct = 0
    fourchan = 0
    onechan = 0
    otherwrong = 0

    for clsname in CLASSES_itos:
        imgcount = 0
        logging.info('Currently at class {}'.format(clsname))
        #image_list = []
        for filenum in range(1, 1+len(os.listdir('./data/{}'.format(clsname)))):
            imgcount+=1
            filename = './data/{}/{}{}.png'.format(clsname, clsname, filenum)

            if imgcount % 1000 == 1:
                logging.info('--at {}st image'.format(imgcount))

            im = Image.open(filename)

            tensor = transform(im)
            if tensor.shape == (3,128,128):
                correct += 1
            elif tensor.shape == (4,128,128):
                fourchan += 1
            elif tensor.shape == (1,128,128):
                onechan += 1
            else:
                otherwrong +=1

    print('correct:{}, fourchan:{}, onechan:{}, otherwrong:{}'.format(correct, fourchan, onechan, otherwrong))




# Function based off
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(fname, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        path: String to the output path of confusion matrix plot
        cm: A NumPy array denoting the confusion matrix
        classes: A list of strings denoting the name of the classes
        normalize: Boolean whether to normalize the confusion matrix or not
        title: String for the title of the plot
        cmap: Colour map for the plot
    """
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix_{}.png".format(fname))

    return


if __name__ == '__main__':
    get_train_val_loader(debug=True)
    #examine_pic_dimensions()
