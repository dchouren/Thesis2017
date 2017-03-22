import sys
from os.path import join
import glob

import matplotlib.pyplot as plt
import pickle




def plot_history(filename, output):
    print(output)
    with open(filename, 'rb') as inf:
        history = pickle.load(inf)

    val_loss = history['val_loss']
    loss = history['loss']

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title(output)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(output)
    plt.clf()


if __name__ == '__main__':
    plot_dir = './plots/loss'

    histories_dir = './histories'
    history_files = glob.glob(join(histories_dir, '*.pickle'))

    for history_file in history_files:
        filebase = history_file.split('/')[-1].split('.')[0]
        plot_history(history_file, join(plot_dir, filebase))
