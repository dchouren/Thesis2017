'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
# from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam, Adamax
from keras import backend as K

from _KDTree import _KDTree
import vision_utils as vutils
from models.utils import _load_model
from gen_utils import meter_distance

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import time
import sys
from os.path import join
import os
import pickle


import ipdb


def euclidean_distance(vects):
    x, y = vects
    distance = K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    if distance == float('Inf'):
        print('inf')
        distance = len(vects)
    return distance


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean((1 - y_true) * K.square(y_pred) + (y_true) * K.square(K.maximum(margin - y_pred, 0)))



# def create_base_network(input_shape):
#     '''Base network to be shared (eq. to feature extraction).
#     '''
#     seq = Sequential()
#     seq.add(Dense(128, input_shape=input_shape, activation='relu'))
#     seq.add(Dropout(0.1))
#     seq.add(Dense(128, activation='relu'))
#     seq.add(Dropout(0.1))
#     seq.add(Dense(128, activation='relu'))
#     return seq


def compute_accuracy(predictions, labels, distance_threshold):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    numeric_preds = 1 * [predictions.ravel() < distance_threshold][0]
    return accuracy_score(numeric_preds, labels)
    # return labels[predictions.ravel() < 0.5].mean()


# create_pairs('2014', '01', 'resources/pairs/2014')

def create_base_network(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2))
    return model


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)



def main():
    launch_time = str(time.time())

    model_name = sys.argv[1]
    data_dir = sys.argv[2]
    nb_epoch = int(sys.argv[3])
    initial_split = int(sys.argv[4])
    optimizer = sys.argv[5]

    identifier = model_name + '_' + str(nb_epoch) + '_' + str(initial_split) + '_' + optimizer


    data_files = sorted(os.listdir(data_dir))[:initial_split]

    input_shape = (3, 224, 224)

    lower_slice = 0
    upper_slice = 1000
    all_pairs = np.empty((len(data_files) * 1000, 2, *input_shape), dtype=np.float32)
    for data_file in data_files:
        new_data = np.squeeze(np.load(join(data_dir, data_file)))
        upper_slice = lower_slice + len(new_data)
        pairs = all_pairs[lower_slice:upper_slice]
        pairs[:] = new_data

        lower_slice = upper_slice

    # pairs = np.vstack([np.squeeze(np.load(join(data_dir, x))) for x in data_files])
    train_split = int(0.9 * len(all_pairs))
    if train_split % 2 == 1:
        train_split += 1
    tr_pairs = all_pairs[:train_split]
    val_pairs = all_pairs[train_split:]
    tr_y = np.asarray([1,0] * int(len(tr_pairs)/2))
    val_y = np.asarray([1,0] * int(len(val_pairs)/2))


    # tr_pairs = np.squeeze(np.load('/tigress/dchouren/thesis/resources/pairs/2014/pairs_10.npy'))
    # nb_epoch = 1

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # X_train = X_train.reshape(60000, 784)
    # X_test = X_test.reshape(10000, 784)
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255
    # input_dim = 784
    # # create training+test positive and negative pairs
    # digit_indices = [np.where(y_train == i)[0] for i in range(10)]
    # tr_pairs, tr_y = create_pairs(X_train, digit_indices)

    # digit_indices = [np.where(y_test == i)[0] for i in range(10)]
    # te_pairs, te_y = create_pairs(X_test, digit_indices)
# input_shape = (784,)

    if model_name == 'base':
        base_network = create_base_network(input_shape)
    else:
        base_network = _load_model(model_name, include_top=True, weights=None)

    print('Base network created')

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(input=[input_a, input_b], output=distance)

    # train
    rms = RMSprop()
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

    optimizer_map = {'rms': rms, 'adadelta': adadelta, 'adagrad': adagrad, 'adam': adam, 'adamax': adamax, 'nadam': nadam}

    opt = optimizer_map[optimizer]
    model.compile(loss=contrastive_loss, optimizer=opt)

    print('Model compiled')

    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y), batch_size=8, nb_epoch=nb_epoch)

    distance_threshold = 0.5

    # compute final accuracy on training and test sets
    tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_pred, tr_y, distance_threshold)
    np.savetxt('/tigress/dchouren/thesis/preds/' + identifier + '_tr', tr_pred)

    # ipdb.set_trace()
    # print(pred)

    val_pred = model.predict([val_pairs[:, 0], val_pairs[:, 1]])
    val_acc = compute_accuracy(val_pred, val_y, distance_threshold)
    np.savetxt('/tigress/dchouren/thesis/preds/' + identifier + '_te', val_pred)


    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * val_acc))

    model.save('/tigress/dchouren/thesis/trained_models/' + identifier)

    # print(history.history.keys())

    with open('/tigress/dchouren/thesis/histories/' + identifier + '.pickle', 'wb') as outf:
        pickle.dump(history.history, outf)

    print('Saved all')


    # list all data in history
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(join('plots', launch_time + '_acc'))
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(join('plots', launch_time + '_acc'))


if __name__ == '__main__':
    sys.exit(main())











