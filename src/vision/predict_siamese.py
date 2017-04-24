import sys
from os.path import join
import os
import time

import h5py
import numpy as np

from keras.models import load_model
from keras import backend as K
K.set_image_data_format('channels_first')

from models.utils import _load_model
from extract_bottlenecks import save_bottleneck_features
from siamese_network import compute_accuracy, build_siamese_network

import ipdb


model_name = sys.argv[1]
optimizer_name = sys.argv[2]
weights = sys.argv[3]

# identifier = weights.split('.')[0]


img_size = (224,224)
batch_size = 32

input_shape = (3,224,224)
model_dir = '/tigress/dchouren/thesis/trained_models/'

weights_path = join(model_dir, weights)

model = build_siamese_network(model_name, input_shape, optimizer_name)

model.load_weights(weights_path, by_name=True)

start_time = time.time()

with h5py.File('/tigress/dchouren/thesis/evaluation/pairs.h5', 'r') as eval_file:
    pairs = eval_file['pairs']
    # preds = save_bottleneck_features(model, year, month, pred_output)
    # preds = model.predict([np.swapaxes(pairs[:,0], 1, 3), np.swapaxes(pairs[:,1], 1, 3)])
    preds = model.predict([pairs[:,0], pairs[:,1]])
    labels = [1,0] * int(len(pairs)/2)
    print('Accuracy: {}'.format(compute_accuracy(preds, labels, 0.5)))

eval_preds = preds.ravel()
pair_preds = zip(eval_preds[::2], eval_preds[1::2])
accuracy = [1 if pair_pred[0] < pair_pred[1] else 0 for pair_pred in pair_preds]

print(sum(accuracy) / len(accuracy))

preds = np.array(preds)
np.save('/tigress/dchouren/thesis/evaluation/preds/{}.npy'.format(weights), accuracy)


# print('Predictions: {}'.format(pred_output))
print('{} seconds'.format(int(time.time() - start_time)))




