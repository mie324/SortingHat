import matplotlib.pyplot as plt


def plot_graph(path, steps, train_data, val_data):
    plt.figure()
    type_title = "Training and Validation Acc."
    plt.title(type_title)
    plt.plot(steps, train_data, label="Train")
    plt.plot(steps, val_data, label="Validation")
    plt.xlabel("epoch num")
    plt.ylabel(type_title)
    plt.legend(loc='best')
    plt.savefig("{}.png".format(path))