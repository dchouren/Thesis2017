'''
Usage: python bottleneck_train.py [model] [task] [dataset] [nb_epoch] [batch_size] [mapping]
Example: python src/vision/bottleneck_train.py vgg16 thumbnail dslr 100 256 dslr

mapping should align with what mapping to use for predictions

See https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
'''


import os
from os.path import join
import sys
import numpy as np
import pickle
import getpass

from models.utils import _load_model
from _utils import mkdir_p

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, SGD

import ipdb


def save_bottleneck_features(model, directory, img_size, batch_size, nb_samples, output_path):
    datagen = ImageDataGenerator(rescale=1. / 255)

    print('Running {}'.format(directory))
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False)
    bottleneck_features = model.predict_generator(generator, nb_samples)
    np.save(open(output_path, 'wb'), bottleneck_features)


def build_top_model(input_shape, nb_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape), name='flatten')
    # model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'), name='fc1')
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'), name='fc2')
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'), name='predictions')

    return model


if len(sys.argv) != 7:
    print (__doc__)
    sys.exit(0)

scratch_dir = join('/spindle', getpass.getuser())

model_name = sys.argv[1]
task = sys.argv[2]
dataset = sys.argv[3]
data_dir = join(scratch_dir, 'images', task, dataset)
resources_dir = join(scratch_dir, 'resources', task, dataset)
if not os.path.exists(resources_dir):
    mkdir_p(resources_dir)
nb_epoch = int(sys.argv[4])
batch_size = int(sys.argv[5])
mapping = sys.argv[6]

img_size = (224,224)

train_data_dir = join(data_dir, 'train')
validation_data_dir = join(data_dir, 'val')

labels = sorted(os.listdir(train_data_dir))
print('Labels: {}'.format(labels))
nb_classes = len(labels)
train_class_sizes = [len(os.listdir(join(train_data_dir, label))) for label in labels]
val_class_sizes = [len(os.listdir(join(validation_data_dir, label))) for label in labels]
nb_train_samples = sum(train_class_sizes)
print('Num train samples: {}'.format(nb_train_samples))
nb_val_samples = sum(val_class_sizes)
class_weights_list = [nb_train_samples / class_size for class_size in train_class_sizes]
class_weights = {}
for i, weight in enumerate(class_weights_list):
    class_weights[i] = weight
print('Class weights: {}'.format(class_weights))



bottlenecks_dir = join(resources_dir, 'bottlenecks')
if not os.path.exists(bottlenecks_dir):
    mkdir_p(bottlenecks_dir)
bf_train_path = join(bottlenecks_dir, 'bottleneck_train_{}_{}_{}_{}_{}.npy'.format(model_name, task, dataset, nb_epoch, mapping))
bf_val_path = join(bottlenecks_dir, 'bottleneck_val_{}_{}_{}_{}_{}.npy'.format(model_name, task, dataset, nb_epoch, mapping))

models_dir = join(resources_dir, 'trained_models')
if not os.path.exists(models_dir):
    mkdir_p(models_dir)
top_model_weights_path = join(models_dir, 'top_{}_{}_{}_{}_{}.h5'.format(model_name, task, dataset, nb_epoch, mapping))
# top_model_weights_path = join(models_dir, top_model_name)

histories_dir = join(resources_dir, 'histories')
if not os.path.exists(histories_dir):
    mkdir_p(histories_dir)
top_model_history_path = join(histories_dir, 'top_{}_{}_{}_{}_{}.pickle'.format(model_name, task, dataset, nb_epoch, mapping))



def train_top_model():
    print('Training top model')
    train_data = np.load(open(bf_train_path, 'rb'))
    validation_data = np.load(open(bf_val_path, 'rb'))

    train_labels = []
    validation_labels = []
    for i, label in enumerate(labels):
        categorical_label = [0] * nb_classes
        categorical_label[i] = 1
        train_labels.extend([categorical_label] * len(os.listdir(join(train_data_dir, label))))
        validation_labels.extend([categorical_label] * len(os.listdir(join(validation_data_dir, label))))
    train_labels = np.asarray(train_labels)
    validation_labels = np.asarray(validation_labels)

    top_model = build_top_model(train_data.shape[1:], nb_classes)

    # ipdb.set_trace()

    learning_rate = 0.001
    rmsprop = RMSprop(lr=learning_rate)
    top_model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

    top_train_history = top_model.fit(train_data, train_labels, nb_epoch=nb_epoch, batch_size=batch_size, validation_data=(validation_data, validation_labels),)
    # model.save_weights(top_model_weights_path)

    top_model.save(top_model_weights_path)
    print('Saved top model to: {}'.format(top_model_weights_path))
    with open(top_model_history_path, 'wb') as outf:
        pickle.dump(top_train_history.history, outf)
    print('Saved training history to: {}'.format(top_model_history_path))

    print('Done')


print('Loading model')
model = _load_model(model_name, include_top=False)
# print('Saving training features')
# save_bottleneck_features(model, train_data_dir, img_size, batch_size, nb_train_samples, bf_train_path)
# print('Saving validation features')
# save_bottleneck_features(model, validation_data_dir, img_size, batch_size, nb_val_samples, bf_val_path)

train_top_model()



