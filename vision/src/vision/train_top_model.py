'''
Usage: python bottleneck_train.py [model] [task] [dataset] [nb_epoch] [batch_size] [label]
Example: python src/train_top_model.py vgg16 thumbnail flickr 100 256 gfbf

See https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''


import numpy as np

from models.utils import _load_model

import glob
import os
from os.path import join
import sys
import pickle

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, SGD
from keras.constraints import maxnorm

import ipdb


if len(sys.argv) != 7:
    print (__doc__)
    sys.exit(0)

scratch_dir = '/spindle/dchouren'

model_name = sys.argv[1]
task = sys.argv[2]
dataset = sys.argv[3]
data_dir = join(scratch_dir, 'images', task, dataset)
resources_dir = join(scratch_dir, 'resources', task, dataset)
if not os.path.exists(resources_dir):
    os.mkdir(resources_dir)
nb_epoch = int(sys.argv[4])
batch_size = int(sys.argv[5])
label = sys.argv[6]

base_dir = './'

img_size = (224,224)

train_data_dir = join(data_dir, 'train')
validation_data_dir = join(data_dir, 'val')

labels = sorted(os.listdir(train_data_dir))
print(labels)
nb_classes = len(labels)
train_class_sizes = [len(os.listdir(join(train_data_dir, label))) for label in labels]
val_class_sizes = [len(os.listdir(join(validation_data_dir, label))) for label in labels]
nb_train_samples = sum(train_class_sizes)
print(nb_train_samples)
nb_val_samples = sum(val_class_sizes)
class_weights_list = [nb_train_samples / class_size for class_size in train_class_sizes]
class_weights = {}
for i, weight in enumerate(class_weights_list):
    class_weights[i] = weight
print(class_weights)
nb_validation_samples = sum([len(os.listdir(join(validation_data_dir, label))) for label in labels])



bottlenecks_dir = join(resources_dir, 'bottlenecks')
if not os.path.exists(bottlenecks_dir):
    os.mkdir(bottlenecks_dir)
bf_train_path = join(bottlenecks_dir, 'bottleneck_train_{}_{}.npy'.format(model_name, label))
bf_val_path = join(bottlenecks_dir, 'bottleneck_val_{}_{}.npy'.format(model_name, label))

models_dir = join(resources_dir, 'trained_models')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
top_model_weights_path = join(models_dir, 'top_{}_{}_{}_{}.h5'.format(model_name, nb_epoch, batch_size, label))

histories_dir = join(resources_dir, 'histories')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
top_model_history_path = join(histories_dir, 'top_{}_{}_{}_{}.pickle'.format(model_name, nb_epoch, nb_train_samples, label))


def build_top_model(input_shape):
    model = Sequential()
    # ipdb.set_trace()
    # model.layers.pop()
    model.add(Flatten(input_shape=input_shape))
    # model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))

    return model


def train_top_model():
    print('Training top model')

    # ipdb.set_trace()

    train_data = np.load(open(bf_train_path, 'rb'))
    validation_data = np.load(open(bf_val_path, 'rb'))

    train_labels = []
    validation_labels = []
    for i, label in enumerate(labels):
        categorical_label = [0] * nb_classes
        categorical_label[i] = 1
        train_labels.extend([categorical_label] * len(os.listdir(join(train_data_dir, label))))
        validation_labels.extend([categorical_label] * len(os.listdir(join(validation_data_dir, label))))
        # validation_labels.extend([i] * (nb_validation_samples // nb_classes))
    train_labels = np.asarray(train_labels)
    validation_labels = np.asarray(validation_labels)

    model = build_top_model(train_data.shape[1:])

    learning_rate = 0.001
    rmsprop = RMSprop(lr=learning_rate)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

    top_train_history = model.fit(train_data, train_labels, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(validation_data, validation_labels),)
    # model.save_weights(top_model_weights_path)

    model.save(top_model_weights_path)
    print('Saved model to: {}'.format(top_model_weights_path))
    with open(finetuned_history_path, 'wb') as outf:
        pickle.dump(top_train_history, outf)
    print('Saved training history to: {}'.format(finetuned_history_path))
    # ipdb.set_trace()


# save_bottleneck_features()
train_top_model()

print('Done')