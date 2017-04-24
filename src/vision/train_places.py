'''
./src/generate_slurm.sh 48:00:00 62GB "python /tigress/dchouren/thesis/src/vision/train_places.py resnet50_50_4600_nadam_2015_weights.23-0.06.h5 nadam" train_places true dchouren@princeton.edu /tigress/dchouren/thesis
'''

import sys
from os.path import join

import h5py
import numpy as np
import pickle

from keras.models import load_model
from keras.models import Model
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from models.utils import _load_model
from extract_bottlenecks import save_bottleneck_features
from siamese_network import compute_accuracy, build_siamese_network, get_optimizer


import ipdb




model_weights = sys.argv[1]
optimizer_name = sys.argv[2]
places_trained_weights = sys.argv[3]
# identifier = sys.argv[3]


weights_dir = '/tigress/dchouren/thesis/trained_models'
weights_path = join(weights_dir, model_weights)

base_dir = '/tigress/dchouren/thesis/resources/places365/images/'

train_data_dir = join(base_dir, 'train')
val_data_dir = join(base_dir, 'val')

epochs = 50
batch_size = 32

nb_train_samples = 1803460
nb_val_samples = 36500



input_shape = (3,224,224)
img_height, img_width = 224, 224


base_cnn = ResNet50(include_top=False, weights=None, input_shape=input_shape)
base_cnn.load_weights(weights_path, by_name=True)
last = base_cnn.output

print('Model weights loaded')

top_model = Flatten()(last)
top_model = Dense(1024, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
preds = Dense(365, activation='softmax')(top_model)

model = Model(base_cnn.input, preds)

for layer in model.layers[:191]:
    layer.trainable = False

optimizer = get_optimizer(optimizer_name)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print('Model compiled')


train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='categorical', shuffle=True)
f = h5py.File('/tigress/dchouren/thesis/resources/places365/images/val.h5')
val_images = f['images']
val_labels = f['labels']

print('Training')
model_dir = '/tigress/dchouren/thesis/trained_models'
save_weights_path = join(model_dir, 'places_' + model_weights + '_{epoch:02d}-{val_loss:.4f}.h5')
checkpointer = ModelCheckpoint(filepath=save_weights_path, verbose=1, save_best_only=True)

history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples/batch_size, epochs=epochs, validation_data=(val_images, val_labels), callbacks=[checkpointer])

h = history.__dict__
h.pop('model', None)
with open('/tigress/dchouren/thesis/histories/' + 'places_' + model_weights + '.pickle', 'wb') as outf:
    pickle.dump(h, outf)

























