import sys
from os.path import join
import glob

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


def plot_two_history(normal_file, iv_file, output):
    print(output)
    with open(normal_file, 'rb') as inf:
        n_history = pickle.load(inf)
    with open(iv_file, 'rb') as inf:
        i_history = pickle.load(inf)

    if 'history' in n_history.keys():
        n_history = n_history['history']

    n_val_loss = n_history['val_loss']
    n_loss = n_history['loss']

    if 'history' in i_history.keys():
        i_history = i_history['history']

    i_val_loss = i_history['val_loss']
    i_loss = i_history['loss']

    plt.plot(n_loss, label='Method 2 train')
    plt.plot(n_val_loss, label='Method 2 val')
    plt.plot(i_loss, label='Method 1 train')
    plt.plot(i_val_loss, label='Method 1 val')
    # plt.title(output)
    plt.ylabel('log loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.semilogy()
    plt.savefig(output)
    plt.clf()


def plot_history(filename, output):
    print(output)
    with open(filename, 'rb') as inf:
        history = pickle.load(inf)

    if 'history' in history.keys():
        history = history['history']

    val_loss = history['val_loss']
    loss = history['loss']

    plt.plot(loss, label='train')
    plt.plot(val_loss, label='val')
    # plt.title(output)
    plt.ylabel('log loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.semilogy()
    plt.savefig(output)
    plt.clf()


if __name__ == '__main__':
    plot_dir = './plots/histories'

    histories_dir = './histories'
    history_files = glob.glob(join(histories_dir, '*.pickle'))
    history_files = [x for x in history_files if not 'resnet' in x]
    history_files = [x for x in history_files if not 'double' in x]
    history_files = [x for x in history_files if not 'triple' in x]

    history_files = [join(histories_dir, 'new_2013_2014_2015_all.h5_50_5400_nadam_None.pickle')]

    normal = join(histories_dir, '2014_12_32000.h5_50_4000_nadam_None.pickle')
    i = join(histories_dir, 'iv_2014_12_32000.h5_28_5333_nadam.pickle')

    plot_two_history(normal, i, join(plot_dir, '12.png'))

    # for history_file in history_files:
    #     filebase = history_file.split('/')[-1].split('.')[0]
    #     try:
    #         plot_two_history(history_file, join(plot_dir, 'iv_' + history_file.split('/')[-1]), join(plot_dir, filebase))
    #         print(filebase)
    #     except:
    #         pass
