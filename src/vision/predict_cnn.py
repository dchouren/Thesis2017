import sys
from os.path import join
import os
import time

import h5py
import numpy as np

from keras.models import load_model
from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16


from models.utils import _load_model
from extract_bottlenecks import save_bottleneck_features
from siamese_network import compute_accuracy
 
import ipdb 
 

model_name = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]

img_size = (224,224)
batch_size = 32
input_shape = (224,224,3)

model_dir = '/tigress/dchouren/thesis/trained_models/'
pred_dir = join('/tigress/dchouren/thesis/preds/test', model_name)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_output = join(pred_dir, year + '_' + month + '.h5')

start_time = time.time() 
K.set_image_data_format('channels_last')
print(K.image_data_format())    
# model = load_model(join(model_dir, model_name))
model = ResNet50(include_top=False, input_shape=input_shape)


with h5py.File('/tigress/dchouren/thesis/evaluation/images.h5', 'r') as eval_file:
    images = eval_file['images']
    # preds = save_bottleneck_features(model, year, month, pred_output)
    images = np.array(images)
    images = images.swapaxes(1, 3)
    print('Image shape: {}'.format(images.shape))
    preds = model.predict(images)
    # labels = [1,0] * int(len(pairs)/2)
    # print(compute_accuracy(preds, labels, 0.5))

np.save('/tigress/dchouren/thesis/evaluation/resnet50_preds.npy', preds)

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



