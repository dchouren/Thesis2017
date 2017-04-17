'''Train a Siamese MLP on pairs of digits from the MNIST dataset.

It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).

[1] "Dimensionality Reduction by Learning an Invariant Mapping"
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

Gets to 99.5% test accuracy after 20 epochs.
3 seconds per epoch on a Titan X GPU
'''

import random
import time
import sys
from os.path import join
import os
import pickle

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(1337)  # for reproducibility
import h5py

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.layers import Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop, Adadelta, Adagrad, Nadam, Adam, Adamax, SGD
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.applications.resnet50 import ResNet50
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from _KDTree import _KDTree
import vision_utils as vutils
from models.utils import _load_model
from gen_utils import meter_distance


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
    loss = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
    return loss


def compute_accuracy(predictions, labels, distance_threshold=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    numeric_preds = 1 * [predictions.ravel() < distance_threshold][0]
    return accuracy_score(numeric_preds, labels)
    # return labels[predictions.ravel() < 0.5].mean()


def create_base_network(input_shape):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128))
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


def get_training_files(file_dir, start_date, end_date, size):
    files = sorted(os.listdir(file_dir))

    start_file = start_date + '_' + str(size) + '.h5'
    end_file = end_date + '_' + str(size) + '.h5'
    training_files = files[files.index(start_file):files.index(end_file) + 1]

    return training_files


def get_optimizer(optimizer='SGD'):
    clipnorm = 1
    if optimizer == 'sgd':
        opt = SGD(lr=0.0001, momentum=0.9, decay=0.0, nesterov=False, clipnorm=clipnorm)
    elif optimizer == 'rms':
        opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
    elif optimizer == 'adam':
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
    elif optimizer == 'adamax':
        opt = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=clipnorm)
    elif optimizer == 'nadam':
        opt = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipnorm=clipnorm)
    else:
        print('Optimizer not supported, defaulting to SGD')
        opt = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False, clipnorm=clipnorm)

    return opt


def build_siamese_network(model_name, input_shape, optimizer):

    if model_name == 'base':
        base_network = create_base_network(input_shape)
    else:
        # base_network = _load_model(model_name, include_top=True, weights=None)
        base_network = ResNet50(include_top=True, weights=None, input_shape=input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    opt = get_optimizer(optimizer)
    model.compile(loss=contrastive_loss, optimizer=opt)

    print('Model compiled')
    return model


def _generator(filename, batch_size=32, index=0, augment=False):
    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    idg = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, zoom_range=0.05)
    while 1:
        data = pairs[index:index+batch_size]
        if augment:
            data = np.array([[idg.random_transform(x[0]), idg.random_transform(x[1])] for x in data])
        yield [data[:,0], data[:,1]], [1,0]*int(batch_size/2)
        index += batch_size
        index = index % len(pairs)
        # print(index)

    f.close()


def _iter_generator(filename, batch_size=32):
    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    while 1:
        for i in range(int(len(pairs) / batch_size)):
            yield [pairs[i:i+2][:,0], pairs[i:i+2][:,1]], [1,0]

    f.close()



def main():
    launch_time = str(time.time())

    model_name = sys.argv[1]
    optimizer_name = sys.argv[2]
    nb_epoch = int(sys.argv[3])
    n_batch = int(sys.argv[4])
    identifier = sys.argv[5]
    pairs_file = sys.argv[6]

    weights_file = None
    if len(sys.argv) > 7:
        weights_file = sys.argv[7]

    input_shape = (3, 224, 224)
    identifier = model_name + '_' + str(nb_epoch) + '_' + str(n_batch) + '_' + optimizer_name + '_' + identifier


    sys.setrecursionlimit(10000)
    model = build_siamese_network(model_name, input_shape, optimizer_name)

    # # model = load_model('/tigress/dchouren/thesis/resnet50_siamese.h5', custom_objects={'contrastive_loss': contrastive_loss})

    # # model.save('/tigress/dchouren/thesis/resnet50_siamese.h5')

    print('Siamese network built')

    # training_files = get_training_files(data_dir, start_date, end_date, 35200)

    # print(join(data_dir, training_files[0]))
    # f = h5py.File(join(data_dir, training_files[0]))
    # all_pairs = np.array(f['pairs'])
    # # all_pairs.swap_axes(1,3)
    # # lower_slice = 0
    # # upper_slice = 1000
    # # all_pairs = np.empty((len(data_files) * 1000, 2, *input_shape), dtype=np.float32)
    # # for data_file in data_files:
    # #     new_data = np.squeeze(np.load(join(data_dir, data_file)))
    # #     upper_slice = lower_slice + len(new_data)
    # #     pairs = all_pairs[lower_slice:upper_slice]
    # #     pairs[:] = new_data

    # #     lower_slice = upper_slice

    # # pairs = np.vstack([np.squeeze(np.load(join(data_dir, x))) for x in data_files])
    # train_split = int(0.9 * len(all_pairs))
    # if train_split % 2 == 1:
    #     train_split += 1
    # tr_pairs = all_pairs[:train_split]
    # val_pairs = all_pairs[train_split:]
    # tr_y = np.asarray([1,0] * int(len(tr_pairs)/2))
    # val_y = np.asarray([1,0] * int(len(val_pairs)/2))


    # history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_y), batch_size=batch_size, epochs=nb_epoch)
    # left = HDF5Matrix('/tigress/dchouren/thesis/resources/pairs/2014_01_35200.h5', 'left')
    # right = HDF5Matrix('/tigress/dchouren/thesis/resources/pairs/2014_01_35200.h5', 'left')
    # history = model.fit([left, right], [1,0]*int(left.shape[0]/2), validation_split=0.1)
    batch_size = 32


    model_dir = '/tigress/dchouren/thesis/trained_models'

    if weights_file:
        model.load_weights(join(model_dir, weights_file))

    save_weights_path = join(model_dir, identifier + '_weights.{epoch:02d}-{val_loss:.2f}.h5')
    checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True)

    n_train_batch = int(n_batch / 10 * 9)

    pairs_file = join('/tigress/dchouren/thesis/resources/pairs/', pairs_file)
    generator = _generator(pairs_file, batch_size=batch_size)
    val_generator = _generator(pairs_file, batch_size=batch_size, index=n_train_batch*batch_size)
    print('Generator constructed')
    history = model.fit_generator(generator, steps_per_epoch=n_train_batch, epochs=nb_epoch, validation_data=val_generator, validation_steps=n_batch-n_train_batch, callbacks=[checkpointer])
    print('Finished fitting')

    distance_threshold = 0.5

    model_save_path = join(model_dir, identifier + '.h5')
    model.save_weights(model_save_path)

    # base_cnn = _load_model(model_name, include_top=True, weights=None)
    # base_cnn = ResNet50(include_top=True, weights=None, input_shape=input_shape)

    # base_cnn.load_weights(save_weights_path, by_name=True)
    # base_cnn.save(join(model_dir, 'base_cnn', identifier +'.h5'))

    # model.save_weights('/tigress/dchouren/thesis/trained_models/' )

    # print(history.history.keys())
    h = history.__dict__
    h.pop('model', None)
    with open('/tigress/dchouren/thesis/histories/' + identifier + '.pickle', 'wb') as outf:
        pickle.dump(h, outf)

    print('Saved')


    # compute final accuracy on training and test sets
    # tr_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    # tr_acc = compute_accuracy(tr_pred, tr_y, distance_threshold)
    # np.savetxt('/tigress/dchouren/thesis/preds/tr/' + identifier + '_tr', tr_pred)
    f = h5py.File('/tigress/dchouren/thesis/resources/pairs/2013_08_35200.h5')
    val_pairs = f['pairs'][:10000]
    val_y = [1,0] * int(len(val_pairs)/2)

    val_pred = model.predict([val_pairs[:, 0], val_pairs[:, 1]])
    val_acc = compute_accuracy(val_pred, val_y, distance_threshold)
    np.savetxt('/tigress/dchouren/thesis/preds/val/' + identifier + '_val', val_pred)

    # print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on validation set: %0.2f%%' % (100 * val_acc))


    print('Total time: {}'.format(str(time.time() - float(launch_time))))


if __name__ == '__main__':
    sys.exit(main())











