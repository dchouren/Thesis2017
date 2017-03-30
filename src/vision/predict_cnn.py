import sys
from os.path import join
import os
import time

import h5py
import numpy as np

from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_first')
from keras.applications.resnet50 import ResNet50


from models.utils import _load_model
from extract_bottlenecks import save_bottleneck_features
from siamese_network import compute_accuracy

import ipdb


model_name = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]

img_size = (224,224)
batch_size = 32

model_dir = '/tigress/dchouren/thesis/trained_models/'
pred_dir = join('/tigress/dchouren/thesis/preds/test', model_name)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_output = join(pred_dir, year + '_' + month + '.h5')

start_time = time.time()
# model = load_model(join(model_dir, model_name))
model = ResNet50(include_top=False, weights='imagenet')

with h5py.File('/tigress/dchouren/thesis/evaluation/images.h5', 'r') as eval_file:
    pairs = eval_file['pairs']
    # preds = save_bottleneck_features(model, year, month, pred_output)
    preds = model.predict(pairs)
    labels = [1,0] * int(len(pairs)/2)
    print(compute_accuracy(preds, labels, 0.5))

np.save('/tigress/dchouren/thesis/evaluation/resnet50.h5', preds)

<<<<<<< HEAD
with h5py.File('/tigress/dchouren/thesis/evaluation/images.h5', 'r') as eval_file:
    pairs = eval_file['pairs']
    # preds = save_bottleneck_features(model, year, month, pred_output)
    preds = model.predict(pairs)
    labels = [1,0] * int(len(pairs)/2)
    print(compute_accuracy(preds, labels, 0.5))



# print('Predictions: {}'.format(pred_output))
print('{} seconds'.format(int(time.time() - start_time)))

ipdb.set_trace()
=======

# print('Predictions: {}'.format(pred_output))
print('{} seconds'.format(int(time.time() - start_time)))

# ipdb.set_trace()
>>>>>>> 443feb688a883f1c977ff58909dc40ae22b03852



