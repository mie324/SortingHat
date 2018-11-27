import torch.optim as optim
import torch.nn as nn
from nlp_model import NLP
from util import *
from result_visualization import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    epochs = 10
    model = NLP()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
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


    steps = np.arange(1, epochs + 1)
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
