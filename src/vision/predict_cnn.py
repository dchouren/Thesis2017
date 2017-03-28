import sys
from os.path import join
import os
import time

from keras.models import load_model

from models.utils import _load_model
from extract_bottlenecks import save_bottleneck_features

import ipdb


model_name = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]

img_size = (224,224)
batch_size = 32

model_dir = '/tigress/dchouren/thesis/trained_models/base_cnn'
pred_dir = join('/tigress/dchouren/thesis/preds/test', model_name)
if not os.path.exists(pred_dir):
    os.makedirs(pred_dir)
pred_output = join(pred_dir, year + '_' + month + '.h5')

start_time = time.time()
model = load_model(join(model_dir, model_name))

preds = save_bottleneck_features(model, year, month, pred_output)

print('Predictions: {}'.format(pred_output))
print('{} seconds'.format(int(time.time() - start_time)))
