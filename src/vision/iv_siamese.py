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
from keras.applications.stock_resnet50 import SResNet50
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import Concatenate, concatenate
from keras.layers.advanced_activations import PReLU
from keras import initializers

from _KDTree import _KDTree
import vision_utils as vutils
from models.utils import _load_model
from gen_utils import meter_distance
from siamese_network import euclidean_distance, eucl_dist_output_shape, get_optimizer, contrastive_loss, create_base_network, _generator, compute_accuracy
from double_siamese import build_variant_model, build_invariant_model, blend_models




import ipdb



def build_iv_siamese_network(input_shape, invariant_model_name='i_resnet50', invariant_model_weights=None, optimizer='nadam'):

    variant_model = build_variant_model(input_shape, 'ResNet', model_weights=None, name='variant', frozen=False)
    invariant_model = build_invariant_model(input_shape, invariant_model_name, invariant_model_weights, frozen=True)

    # ipdb.set_trace()

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    variant_processed_a = variant_model(input_a)
    invariant_processed_a = invariant_model(input_a)
    variant_processed_b = variant_model(input_b)
    invariant_processed_b = invariant_model(input_b)


    processed_a = concatenate([variant_processed_a, invariant_processed_a])
    processed_b = concatenate([variant_processed_b, invariant_processed_b])

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    opt = get_optimizer(optimizer)
    model.compile(loss=contrastive_loss, optimizer=opt)


    print('Model compiled')
    return model



def ranking_accuracy(predictions):
    pair_preds = zip(predictions[::2], predictions[1::2])

    accuracy = [1 if pair_pred[0] < pair_pred[1] else 0 for pair_pred in pair_preds]

    return sum(accuracy) / len(accuracy)





def main():
    launch_time = str(time.time())

    model_name = sys.argv[1]
    optimizer_name = sys.argv[2]
    nb_epoch = int(sys.argv[3])
    # n_batch = int(sys.argv[4])
    pairs = sys.argv[4]
    identifier = 'iv'
    if len(sys.argv) > 5:
        identifier = sys.argv[5]

    weights_file = None
    if len(sys.argv) > 6:
        weights_file = sys.argv[6]

    input_shape = (3, 224, 224)

    sys.setrecursionlimit(10000)
    weights = None
    model = build_iv_siamese_network(input_shape, optimizer=optimizer_name)

    print('Siamese network built')

    # ipdb.set_trace()

    batch_size = 6
    pairs_file = join('/tigress/dchouren/thesis/resources/pairs/', pairs)
    print(pairs_file)
    n_batch = 0
    with h5py.File(pairs_file, 'r') as f:
        x = f['pairs']
        n_batch = int(x.shape[0] / batch_size)
    print(n_batch)

    identifier = identifier + '_' + pairs + '_' + str(nb_epoch) + '_' + str(n_batch) + '_' + optimizer_name


    model_dir = '/tigress/dchouren/thesis/trained_models'

    if weights_file:
        model.load_weights(join(model_dir, weights_file))

    save_weights_path = join('/tigress/dchouren/thesis/trained_models', identifier + '_weights.{epoch:02d}-{val_loss:.4f}.h5')
    checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True)

    print(save_weights_path)

    n_train_batch = int(n_batch / 10 * 9)

    augment = False
    generator = _generator(pairs_file, batch_size=batch_size, augment=augment)
    val_generator = _generator(pairs_file, batch_size=batch_size, index=n_train_batch*batch_size, augment=augment)
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


