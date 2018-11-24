import torch.optim as optim
import torch.nn as nn
import argparse
import time
from util import *
from datetime import datetime
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):
    train_acc = np.zeros(args.epochs)
    train_loss = np.zeros(args.epochs)
    val_acc = np.zeros(args.epochs)
    val_loss = np.zeros(args.epochs)

    # Data processing
    train_loader, val_loader = get_train_val_loader(
        args.batch_size, debug=args.debug, fourclass=args.fourclass)
    logging.info('loaders loaded')

    # ==============+++++++++++===============+++++++++++===============+++++++++++===============+++++++++++
    # ==============+++++++++++===============+++++++++++===============+++++++++++===============+++++++++++
    # Transfer learning stuff
    import torchvision.models as models
    inception = models.inception_v3(pretrained=True)
    for param in inception.parameters():
        param.requires_grad = False
    num_inputs = inception.fc.in_features
    inception.fc = nn.Linear(num_inputs, 4 if args.fourclass else 11)
    # ==============+++++++++++===============+++++++++++===============+++++++++++===============+++++++++++
    # ==============+++++++++++===============+++++++++++===============+++++++++++===============+++++++++++

    model = inception.to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    logging.info('good sofar')
    start_time = time.time()
    for epoch in range(args.epochs):
        logging.info('epoch {} ...'.format(epoch+1))
        accum_loss = 0.0
        num_trained = 0
        tot_corr = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(inputs)
            #print(predictions.shape)
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
        valid_acc, valid_loss = evaluate(model.eval(), val_loader, loss_fcn, epoch, fourclass=args.fourclass)
        model = model.train()
        train_acc[epoch] = tot_corr * 100 / num_trained
        train_loss[epoch] = accum_loss / (i + 1)
        val_acc[epoch] = valid_acc*100
        val_loss[epoch] = valid_loss

        # print('epoch: %d, loss: %f, training acc: %f%%, validation acc: %f%%' %
        #       (epoch + 1, train_loss[epoch], train_acc[epoch], val_acc[epoch]))
        logging.info('epoch %d, loss: %f, training acc: %f%%, validation acc: %f%%' %
              (epoch + 1, train_loss[epoch], train_acc[epoch], val_acc[epoch]))

    print('Finished training:\n',
          'train accuracy:', max(train_acc), 'train loss:', min(train_loss), '\n',
          'validation accuracy:', max(val_acc), 'validation loss:', min(val_loss), '\n')

    # plotting related
    time_elapsed = time.time() - start_time
    print("time elapsed:", time_elapsed)

    now = datetime.now()
    torch.save(model, 'model_{:02d}{:02d}_{:02d}{:02d}.pt'.format(now.month, now.day, now.hour, now.minute))

    #logging.info('generating confusion matrix')


def evaluate(model, val_loader, loss_fcn, epoch, gen_conf_mat=True, fourclass=False):
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

    if gen_conf_mat and epoch%10==9:
        cm = confusion_matrix(y_true, y_pred)
        cls_itos = CLASSES4_itos if fourclass else CLASSES_itos
        plot_confusion_matrix('debug{}'.format(epoch+1), cm, cls_itos)

    length = len(val_loader.dataset)

    return float(total_corr)/length, float(accum_loss/(j+1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--fourclass', type=int, default=0)

    args = parser.parse_args()
    main(args)
