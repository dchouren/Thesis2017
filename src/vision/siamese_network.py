'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

from _KDTree import _KDTree
import vision_utils as vutils
from os.path import join

from gen_utils import meter_distance

from models.utils import _load_model

import time
import sys


import ipdb


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)



def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))



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


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# create_pairs('2014', '01', 'resources/pairs/2014')

def create_base_network(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(1))
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



nb_epoch = int(sys.argv[1])

tr_pairs = np.squeeze(np.load('/tigress/dchouren/thesis/resources/pairs/2014/pairs_10.npy'))
# tr_pairs = np.vstack((np.squeeze(np.load('resources/pairs/2014/pairs_0.npy')), np.squeeze(np.load('resources/pairs/2014/pairs_1.npy'))))
tr_y = np.asarray([1,0] * int(len(tr_pairs)/2))
te_pairs = np.squeeze(np.load('/tigress/dchouren/thesis/resources/pairs/2014/pairs_1.npy'))[:100]
te_y = np.asarray([1,0] * int(len(te_pairs)/2))
input_dim = 784
# nb_epoch = 1

input_shape = (3, 224, 224)

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


base_network = create_base_network(input_shape)

# base_network = _load_model('resnet50', include_top=True, weights=None)

print('Base network created')

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)

print('Model compiled')

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), batch_size=8, nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)

pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))













