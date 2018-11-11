import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.signal as signal
import numpy as np
import argparse
import glob

from PIL import Image
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import CNN


classes = ('snackpackage', 'juicebox', 'coffeecup', 'togobox', 'popcan', 'glassbottle',
           'plasticbottle', 'perishable', 'plasticbag', 'newspaper')

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Data processing
data =



def main(args):
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(CNN.parameters(), lr=args.lr)

    train_err = np.zeros(args.num_epochs)
    train_loss = np.zeros(args.num_epochs)
    val_err = np.zeros(args.num_epochs)
    val_loss = np.zeros(args.num_epochs)


    dataset =
    trainset, valset =

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    for epoch in range(args.num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = cnn(inputs)
            loss = loss_fcn(outputs, labels)

            # magic
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % args.eval_every == args.eval_every-1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--eval_every', type=int, default=64)
    parser.add_argument('--kernel-size', type=int, default=5)
    parser.add_argument('--num-kernels', type=int, default=50)

    args = parser.parse_args()

    main(args)
