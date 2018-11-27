import torch.optim as optim
import torch.nn as nn
from nlp_model import NLP
from util import *
from result_visualization import *
import pandas as pd
from dataset import NLPDataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_loaders(bs):
    df = pd.read_csv('./data/waste_wizard.csv')
    df = df[df['label'] != 4]
    dftrain = df[df.index %5 != 0]
    dfval = df[df.index %5 == 0]
    train_loader = DataLoader(NLPDataset(dftrain['words'], dftrain['label']), batch_size=bs, shuffle=True)
    val_loader = DataLoader(NLPDataset(dfval['words'], dfval['label']), batch_size=bs, shuffle=False)
    return train_loader, val_loader

def main():
    bs = 16
    epochs = 10
    lr = 0.01
    train_loader, val_loader = get_loaders(bs)


    model = NLP()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_acc = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    val_acc = np.zeros(epochs)
    val_loss = np.zeros(epochs)
    for epoch in range(epochs):
        accum_loss = 0.0
        num_trained = 0
        tot_corr = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fcn(predictions, labels.long())

            # magic
            loss.backward()
            optimizer.step()

            # count correct predictions
            accum_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            corr = (predicted == labels.long()).sum().item()
            tot_corr += corr

            num_trained += len(labels)

        # Print statistics
        valid_acc, valid_loss = evaluate(model.eval(), val_loader, loss_fcn)

        logging.info('epoch %d, loss: %f, training acc: %f%%, validation acc: %f%%' %
                     (epoch + 1, train_loss[epoch], train_acc[epoch], val_acc[epoch]))

    print('Finished training:\n',
          'train accuracy:', max(train_acc), 'train loss:', min(train_loss), '\n',
          'validation accuracy:', max(val_acc), 'validation loss:', min(val_loss), '\n')

    import re
    import time
    steps = np.arange(1, epochs + 1)
    localtime = time.asctime(time.localtime(time.time()))
    path = 'nlp_main_' + re.sub(r':', '-', localtime[11:19])
    plot_graph(path, steps, train_acc, val_acc)


def evaluate(model, val_loader, loss_fcn):
    total_corr = 0
    accum_loss = 0
    y_true, y_pred = [], []

    for j, data in enumerate(val_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        predictions = model(inputs)
        _ , predicted = torch.max(predictions, 1)
        loss = loss_fcn(predictions, labels.long())

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

        corr = (predicted == labels.long()).sum().item()
        total_corr += corr

        accum_loss += loss.item()

    length = len(val_loader.dataset)

    return float(total_corr)/length, float(accum_loss/(j+1))


if __name__ == "__main__":
    main()
