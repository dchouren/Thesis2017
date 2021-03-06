import random
import time
import sys
from os.path import join
import os
import pickle
import glob

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
from siamese_network import euclidean_distance, eucl_dist_output_shape, get_optimizer, contrastive_loss, create_base_network, _generator


import ipdb


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    popped_layer = model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
    return popped_layer


def build_variant_model(input_shape, model_name, model_weights=None, name='variant', frozen=False):

    if model_name == 'base':
        model = create_base_network(input_shape)
    else:
        model = ResNet50(include_top=False, weights=None, input_shape=input_shape, name=name)

        model = Model(model.input, Flatten()(model.output))
        # model = Flatten()(model.output)
        # pop_layer(model)
    if model_weights:
        print('loading weights')
        model.load_weights(model_weights, by_name=True)

    if frozen:
        for layer in model.layers:
            layer.trainable = False

    return model


def build_invariant_model(input_shape, model_name, model_weights=None, frozen=False):

    if model_name == 'base':
        model = create_base_network(input_shape)
        model.load_weights(model_weights, by_name=True)
    else:
        model = SResNet50(include_top=False, weights=model_weights, input_shape=input_shape)

        flat = Flatten()(model.output)
        # softmax = Dense(1000, activation='softmax')(flat)
        # output = concatenate([flat, softmax])
        output = flat
        model = Model(model.input, output)


        # model = Flatten()(model)
        # pop_layer(model)
        # pop_layer(model)

        if frozen:
            for layer in model.layers:
                layer.trainable = False

    return model


def skip_invariant(models, layer_size_1, layer_size_2, name):
    variant = Dense(layer_size_1, name=name + '_blend_1')(models[1])
    variant = PReLU(weights=None, alpha_initializer='zero')(variant)
    if layer_size_2 > 0:
        variant = Dropout(0.3)(variant)
        variant = Dense(layer_size_2, name=name + '_blend_2')(variant)
        variant = PReLU(weights=None, alpha_initializer='zero')(variant)

    model = concatenate([models[0], variant])

    return model


# def contrastive_loss(y_true, y_pred):
#     '''Contrastive loss from Hadsell-et-al.'06
#     http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     '''
#     margin = 1
#     loss = K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#     return loss




def blend_models(models, layer_size_1, layer_size_2, name):
    model = concatenate(models)
    model = Dense(layer_size_1, kernel_initializer=initializers.he_normal(), name=name + '_blend_1')(model)
    model = PReLU(weights=None, alpha_initializer='zero')(model)
    if layer_size_2 > 0:
        model = Dropout(0.3)(model)
        model = Dense(layer_size_2,kernel_initializer=initializers.he_normal(), name=name + '_blend_2')(model)
        model = PReLU(weights=None, alpha_initializer='zero')(model)
    #     model = Dropout(0.3)(model)
    #     model = Dense(layer_size_2,kernel_initializer=initializers.he_normal(), name=name + '_blend_3')(model)
    #     model = PReLU(weights=None, alpha_initializer='zero')(model)

    return model


def build_double_siamese_network(input_shape, variant_model_name, variant_model_weights, layer_size_1, layer_size_2, invariant_model_name='resnet50', invariant_model_weights='imagenet', optimizer='nadam'):

    variant_model = build_variant_model(input_shape, variant_model_name, variant_model_weights, frozen=True)
    invariant_model = build_invariant_model(input_shape, invariant_model_name, invariant_model_weights, frozen=True)

    # ipdb.set_trace()

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    variant_processed_a = variant_model(input_a)
    invariant_processed_a = invariant_model(input_a)
    variant_processed_b = variant_model(input_b)
    invariant_processed_b = invariant_model(input_b)


    processed_a = blend_models([variant_processed_a, variant_processed_a, invariant_processed_a], layer_size_1, layer_size_2, name='a')
    processed_b = blend_models([variant_processed_b, variant_processed_b, invariant_processed_b], layer_size_1, layer_size_2, name='b')

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model(inputs=[input_a, input_b], outputs=distance)

    opt = get_optimizer(optimizer)
    model.compile(loss=contrastive_loss, optimizer=opt)


    print('Model compiled')
    return model


def _fold_generator(filename, leave_out_start, leave_out_end, batch_size=32, index=0, augment=False):

    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    total_pairs = pairs.shape[0]

    idg = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, zoom_range=0.05)
    while 1:
        if index == leave_out_start:
            index = leave_out_end

        if index + batch_size > total_pairs:
            data = pairs[index:]
            index = 0
        else:
            data = pairs[index:index+batch_size]
            index += batch_size
            index %= total_pairs

        if augment:
            data = np.array([[idg.random_transform(x[0]), idg.random_transform(x[1])] for x in data])
        yield [data[:,0], data[:,1]], [1,0]*int(len(data)/2)

    f.close()


def ranking_accuracy(predictions):
    pair_preds = zip(predictions[::2], predictions[1::2])

    accuracy = [1 if pair_pred[0] < pair_pred[1] else 0 for pair_pred in pair_preds]

    return sum(accuracy) / len(accuracy)



def main():
    launch_time = str(time.time())

    model_name = sys.argv[1]
    optimizer_name = sys.argv[2]
    nb_epoch = int(sys.argv[3])
    weights_file = sys.argv[4]
    layer_size_1 = int(sys.argv[5])
    layer_size_2 = int(sys.argv[6])
    k = int(sys.argv[7])

    identifier = ''
    if len(sys.argv) > 8:
        identifier = sys.argv[8]


    input_shape = (3, 224, 224)
    identifier = 'double_' + weights_file + '_' + str(layer_size_1) + '_' + str(layer_size_2) + '_' + str(nb_epoch) + '_' + optimizer_name + '_' + identifier


    sys.setrecursionlimit(10000)


    batch_size = 16


    model_dir = '/tigress/dchouren/thesis/trained_models'
    weights = join(model_dir, weights_file)


    pairs_file = '/tigress/dchouren/thesis/evaluation/pairs.h5'
    with h5py.File(pairs_file, 'r') as f:
        num_pairs = f['pairs'].shape[0]
    n_batch = int(num_pairs / batch_size)
    n_train_batch = int(n_batch * 9 / 10)

    skip = int((num_pairs - num_pairs % k*batch_size) / (k*batch_size)) * batch_size
    leave_out_indices = list(np.arange(0, num_pairs, skip))
    leave_out_indices.pop(-1)
    leave_out_indices += [0]
    leave_out_indices = np.array(leave_out_indices)

    print(leave_out_indices)

    accuracies = []

    for i in range(len(leave_out_indices) - 1):
        print('\n\n\nStarting training without fold {}'.format(i))
        model = build_double_siamese_network(input_shape, model_name, weights, layer_size_1, layer_size_2, optimizer=optimizer_name)
        print('Double Siamese network built')

        # ipdb.set_trace()

        # ipdb.set_trace()
        fold_model_dir = join('/tigress/dchouren/thesis/trained_models', identifier, str(i))
        if not os.path.exists(fold_model_dir):
            os.makedirs(fold_model_dir)
        save_weights_path = join(fold_model_dir, '{epoch:02d}-{val_loss:.4f}.h5')
        checkpointer = ModelCheckpoint(save_weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        augment = True
        generator = _fold_generator(pairs_file, leave_out_indices[i], leave_out_indices[i+1], batch_size=batch_size, index=0, augment=augment)
        val_generator = _fold_generator(pairs_file, -1, -1, batch_size=batch_size, index=leave_out_indices[i], augment=augment)
        print('Generator constructed')

        n_train_batch = int(leave_out_indices[i] / batch_size) + int((num_pairs - leave_out_indices[i+1]) / batch_size) + 1
        if leave_out_indices[i+1] == 0:
            n_val_batch = int((num_pairs - leave_out_indices[i]) / batch_size)
        else:
            n_val_batch = int((leave_out_indices[i+1] - leave_out_indices[i]) / batch_size)

        # ipdb.set_trace()    

        history = model.fit_generator(generator, steps_per_epoch=n_train_batch, epochs=nb_epoch, validation_data=val_generator, validation_steps=n_val_batch, callbacks=[checkpointer])



        # n_val_batch = n_batch - n_train_batch

        # augment = True
        # generator = _generator(pairs_file, batch_size=batch_size, augment=augment)
        # val_generator = _generator(pairs_file, batch_size=batch_size, index=n_train_batch*batch_size, augment=augment)
        # print('Generator constructed')
        # history = model.fit_generator(generator, steps_per_epoch=n_train_batch, epochs=nb_epoch, validation_data=val_generator, validation_steps=n_batch-n_train_batch, callbacks=[checkpointer])


        print('Finished fitting')

        k_identifier = identifier + '_' + str(i)
        save_weights_path = join(model_dir, k_identifier + '.h5')
        model.save_weights(save_weights_path)

        h = history.__dict__
        h.pop('model', None)
        with open('/tigress/dchouren/thesis/histories/' + k_identifier + '.pickle', 'wb') as outf:
            pickle.dump(h, outf)

        print('Saved weights and history')

        newest_weights = min(glob.iglob(join(fold_model_dir, '*.h5')), key=os.path.getctime)
        model = load_model(newest_weights, custom_objects={'contrastive_loss': contrastive_loss})

        val_generator = _fold_generator(pairs_file, -1, -1, batch_size=batch_size, index=leave_out_indices[i], augment=False)
        val_predictions = model.predict_generator(val_generator, steps=n_val_batch)
        accuracy = ranking_accuracy(val_predictions)
        accuracies += [accuracy]

        print('Accuracy for fold {}: {}'.format(i, accuracy))

        print('Total time: {}'.format(str(time.time() - float(launch_time))))

    print('Average accuracy: {}'.format(sum(accuracies)/len(accuracies)))


if __name__ == '__main__':
    sys.exit(main())











