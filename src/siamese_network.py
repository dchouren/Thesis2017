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
from keras.optimizers import RMSprop
from keras import backend as K

from _KDTree import _KDTree
import vision_utils as vutils
from os.path import join

from gen_utils import meter_distance

from models.utils import _load_model

import time


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


def create_pos_pairs(dist, KDTree):
    return KDTree.get_pos_pairs(dist)

def create_neg_pairs(dist, KDTree):
    return KDTree.get_neg_pairs(dist)

# def create_pairs(data_file, sample_size):
#     with open(data_file, 'r') as inf:
#         data = inf.readlines()
#     data = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x(.split(',')[2:])] for x in data])

#     sample_data = data[np.random.choice(data.shape[0], sample_size)]
#     K = _KDTree(sample_data)

#     pos_dist = 0.000014
#     neg_dist = 0.1838    
#     pos_pairs = create_pos_pairs(pos_dist, K)
#     neg_pairs = create_neg_pairs(neg_dist, K)

#     n = min(len(pos_pairs), len(neg_pairs))
#     pos_pairs = pos_pairs[:n]
#     neg_pairs = neg_pairs[:n]

#     return pos_pairs, neg_pairs

def create_pairs(year, month, output_dir):
    fpath_base_dir = '/tigress/dchouren/thesis/resources/paths'
    image_base_dir = '/scratch/network/dchouren/images'
    data_file = join(fpath_base_dir, year, month)
    image_dir = join(image_base_dir, year, month, month)

    sample_size = 1000

    with open(data_file, 'r') as inf:
        data = inf.readlines()

    # ipdb.set_trace()
    data_array = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data])

    sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    K = _KDTree(sample_data)

    pos_dist = 0.000014
    neg_dist = 0.1838    
    pos_pairs = create_pos_pairs(pos_dist, K)

    create_pairs_helper(pos_pairs, data_array, image_dir, output_dir)

    # return pairs, labels


def create_pairs_helper(pos_pairs, all_image_data, image_dir, output_dir):

    pairs = []
    labels = []

    # pos_pair_images = []
    # neg_pair_images = []

    # image_dir = '/scratch/network/dchouren/images/2014/01_new/01_new'

    last_time = time.time()

    count = 0

    for i, (base_data, pos_data) in enumerate(pos_pairs):
        base_filename = base_data[2].split('/')[-1]
        base_image = vutils.load_and_preprocess_image(join(image_dir, base_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
        pos_filename = pos_data[2].split('/')[-1]
        pos_image = vutils.load_and_preprocess_image(join(image_dir, pos_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

        possible_neg_image = all_image_data[np.random.choice(all_image_data.shape[0])]
        while meter_distance(possible_neg_image[:2], base_data[:2]) < 2000:
            possible_neg_image = all_image_data[np.random.choice(
                all_image_data.shape[0])]

        neg_filename = possible_neg_image[2].split('/')[-1]
        neg_image = vutils.load_and_preprocess_image(join(image_dir, neg_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

        if base_image is None or pos_image is None or neg_image is None:
            continue

        pos_pair = [base_image, pos_image]
        neg_pair = [base_image, neg_image]

        # ipdb.set_trace()
        pairs += [pos_pair]
        pairs += [neg_pair]
        labels += [1,0]

        if i % 100 == 0:
            print(i, time.time() - last_time)

        if len(labels) == 10000:
            np.save(open(join(output_dir, 'pairs_' + str(count) + '.npy')), pairs)
            labels = []
            pairs = []

    # labels = [1,0] * len(pos_pairs)
    # pairs = zip(pos_pair_images, neg_pair_images)

    # return np.array(pairs), np.array(labels)




# def create_pairs(x, digit_indices):
#     '''Positive and negative pair creation.
#     Alternates between positive and negative pairs.
#     '''
#     pairs = []
#     labels = []
#     n = min([len(digit_indices[d]) for d in range(10)]) - 1
#     for d in range(10):
#         for i in range(n):
#             z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
#             pairs += [[x[z1], x[z2]]]
#             inc = random.randrange(1, 10)
#             dn = (d + inc) % 10
#             z1, z2 = digit_indices[d][i], digit_indices[dn][i]
#             pairs += [[x[z1], x[z2]]]
#             labels += [1, 0]
#     return np.array(pairs), np.array(labels)


def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


# create_pairs('2014', '01', 'resources/pairs/2014')



tr_pairs = np.squeeze(np.load('resources/pairs/2014/pairs_0.npy'))
# tr_pairs = np.vstack((np.squeeze(np.load('resources/pairs/2014/pairs_0.npy')), np.squeeze(np.load('resources/pairs/2014/pairs_1.npy'))))
tr_y = [1,0] * int(len(tr_pairs[0])/2)
te_pairs = np.squeeze(np.load('resources/pairs/2014/pairs_1.npy'))
te_y = [1,0] * int(len(te_pairs[0])/2)
input_dim = 784
nb_epoch = 20


# ipdb.set_trace()

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255
input_dim = 784
nb_epoch = 20

input_shape = (2, 3, 224, 224)

# create training+test positive and negative pairs
# digit_indices = [np.where(y_train == i)[0] for i in range(10)]
# tr_pairs, tr_y = create_pairs(X_train, digit_indices)

# digit_indices = [np.where(y_test == i)[0] for i in range(10)]
# te_pairs, te_y = create_pairs(X_test, digit_indices)


# network definition
# base_network = create_base_network(input_dim)
base_network = _load_model('resnet50', include_top=False)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[input_a, input_b], output=distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms)
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y), batch_size=128, nb_epoch=nb_epoch)

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(pred, tr_y)
pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(pred, te_y)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))