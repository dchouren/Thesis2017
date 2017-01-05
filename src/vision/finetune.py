'''
Usage: python finetune.py [model] [task] [dataset] [top_model] [num_epochs] [samples_per_epoch_factor] [batch_size] [mapping]
Example: python src/vision/finetune.py vgg16 thumbnail flickr full_vgg16_100_256_gfbf.h5 250 2 32 gfbf
python src/vision/finetune.py vgg16 thumbnail dslr full_vgg16_100_55707_dslr.h5 100 8 32 dslr

samples_per_epoch_factor is used to downsample the number of images in your train/val directories. Each epoch will train on ALL the images in these directories, which might be a lot. I usually ran ~15k images per epoch
mapping should align with what mapping to use for predictions
'''

import os
from os.path import join
import sys
import pickle
import h5py
import getpass

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import RMSprop, SGD
from keras.layers import Dense, GlobalAveragePooling2D

import ipdb


if len(sys.argv) != 9:
    print (__doc__)
    sys.exit(0)

scratch_dir = join('/scratch', getpass.getuser())

model_name = sys.argv[1]
task = sys.argv[2]
dataset = sys.argv[3]
top_model_weights_name = sys.argv[4]
nb_epoch = int(sys.argv[5])
samples_per_epoch_factor = int(sys.argv[6])
batch_size = int(sys.argv[7])
mapping = sys.argv[8]



data_dir = join(scratch_dir, 'images', task, dataset)
resources_dir = join(scratch_dir, 'resources', task, dataset)
if not os.path.exists(resources_dir):
    os.mkdir(resources_dir)


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

epoch_size = nb_train_samples / samples_per_epoch_factor

models_dir = join(resources_dir, 'trained_models')
if not os.path.exists(models_dir):
    os.mkdir(models_dir)
top_model_weights_path = join(models_dir, top_model_weights_name)
finetuned_weights_path = join(models_dir, 'finetuned_{}_{}_{}_{}_{}.h5'.format(model_name, task, dataset, nb_epoch, mapping))

histories_dir = join(resources_dir, 'histories')
if not os.path.exists(histories_dir):
    os.mkdir(models_dir)
finetuned_history_path = join(histories_dir, 'finetuned_{}_{}_{}_{}_{}.pickle'.format(model_name, task, dataset, nb_epoch, mapping))

print('Finetuned model will be saved to {}'.format(finetuned_weights_path))


def build_top_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape), name='flatten')
    # model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'), name='fc1')
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'), name='fc2')
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'), name='predictions')

    return model


def finetune_model():

    # this architecture corresponds to VGG16, have not figured out how to load weights for RESNET
    print('Loading vgg16 model')
    img_width, img_height = 224,224
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    f = h5py.File('/scratch/dchouren/resources/weights/imagenet/{}_weights.h5'.format(model_name))
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()

    top_model = build_top_model(model.output._keras_shape[1:])
    top_model.load_weights(top_model_weights_path)

    model.add(top_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical')

    # fine-tune the model
    training_history = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples/samples_per_epoch_factor,
        nb_epoch=nb_epoch,
        max_q_size=1024,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples/samples_per_epoch_factor)

    model.save(finetuned_weights_path)
    print('Saved model to: {}'.format(finetuned_weights_path))
    with open(finetuned_history_path, 'wb') as outf:
        pickle.dump(training_history.history, outf)
    print('Saved training history to: {}'.format(finetuned_history_path))


finetune_model()

print('Done')
