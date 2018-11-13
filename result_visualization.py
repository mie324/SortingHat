import scipy.signal as signal
import matplotlib.pyplot as plt


GROUP = 7
POLYN = 3


def plot_graph(path, train_data, val_data, group=GROUP, polyn=POLYN):
    plt.figure()
    type_title = "Training and Validation Acc."
    plt.title(type_title)
    tr_data = signal.savgol_filter(train_data["train_acc"], group, polyn)
    vl_data = signal.savgol_filter(val_data["val_acc"], group, polyn)
    plt.plot(train_data["steps"], tr_data, label="Train")
    plt.plot(val_data["steps"], vl_data, label="Validation")
    plt.xlabel("epoch num")
    plt.ylabel(type_title)
    plt.legend(loc='best')
    plt.savefig("{}.png".format(path))


def load_csv(config):
    model_path = config
    train_file = './csv/train_acc_{}.csv'.format(model_path)
    val_file = './csv/val_acc_{}.csv'.format(model_path)
    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)

    return train_data, val_data


def plot_config(config, eval_every):
    model_path = config
    train_plot, val_plot = load_csv(config)
    plot_graph(model_path, train_plot, val_plot, eval_every)
