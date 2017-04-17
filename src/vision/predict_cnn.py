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
from vision_utils import mean_center
 
import ipdb 
 

model_name = sys.argv[1]

model_dir = '/tigress/dchouren/thesis/trained_models/base_cnn'

start_time = time.time() 
K.set_image_data_format('channels_first')
model = load_model(join(model_dir, model_name) + '.h5')

with h5py.File('/tigress/dchouren/thesis/evaluation/images.h5', 'r') as eval_file:
    images = eval_file['images']
    # preds = save_bottleneck_features(model, year, month, pred_output)
    images = np.array(images)
    ipdb.set_trace()
    # images = np.array(map(mean_center, images))
    # images = images.swapaxes(1, 3)
    print('Image shape: {}'.format(images.shape))
    preds = model.predict(images)
    # labels = [1,0] * int(len(pairs)/2)
    # print(compute_accuracy(preds, labels, 0.5))

np.save('/tigress/dchouren/thesis/evaluation/preds/{}_preds.npy'.format(model_name), preds)

print('{} seconds'.format(int(time.time() - start_time)))










