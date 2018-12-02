import torch.optim as optim
import torch.nn as nn
from nlp_model import NLP
from util import *
from result_visualization import *
import pandas as pd
import dataset
from gensim.models import Word2Vec
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_loaders(bs):
    df = pd.read_csv('./data/waste_wizard.csv')
    df = df[df['label'] != 4]
    dftrain = df[df.index %5 != 0]
    dfval = df[df.index %5 == 0]
    wv = Word2Vec.load('./data/w2v_wiki300').wv
    train_loader = DataLoader(dataset.NLPDataset(wv, dftrain['words'], dftrain['label']), batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset.NLPDataset(wv, dfval['words'], dfval['label']), batch_size=bs, shuffle=False)
    return train_loader, val_loader

def main():
    bs = 16
    epochs = 40
    lr = 0.001
    train_loader, val_loader = get_loaders(bs)


    model = NLP()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_accs = np.zeros(epochs)

    val_accs = np.zeros(epochs)
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
            # print(predictions)
            loss = loss_fcn(predictions, labels.long())

            # magic
            loss.backward()
            optimizer.step()

            # count correct predictions
            accum_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            # print(predicted)
            corr = (predicted == labels.long()).sum().item()
            tot_corr += corr

            num_trained += len(labels)

        # Print statistics
        valid_acc, valid_loss = evaluate(model.eval(), val_loader, loss_fcn)
        train_accs[epoch]= tot_corr/num_trained
        val_accs[epoch]= valid_acc

        logging.info('epoch %d, training acc: %f%%, validation acc: %f%%' %
                     (epoch + 1, train_accs[epoch], val_accs[epoch]))

    # print('Finished training:\n',
    #       'train accuracy:', max(train_acc), 'train loss:', min(train_loss), '\n',
    #       'validation accuracy:', max(val_acc), 'validation loss:', min(val_loss), '\n')

    import re
    import time
    steps = np.arange(1, epochs + 1)
    localtime = time.asctime(time.localtime(time.time()))
    path = 'nlp_main_' + re.sub(r':', '-', localtime[11:19])
    plot_graph(path, steps, train_accs, val_accs)


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
