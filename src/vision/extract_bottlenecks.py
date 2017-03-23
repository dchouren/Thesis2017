'''
Run images through a topless base model and save features for future predicting/training. You need to put the image directory you want to save in a wrapper directory. Eg, if you want to save san_francisco images, you need to hide your images in $IMAGES/testsets/wrapper/san_francisco, where san_francisco is the only directory in wrapper/.

usage: python extract_bottlenecks.py im_dir output model_name
example: python src/vision/extract_bottlenecks.py $IMAGES/testsets/wrapper $RESOURCES/bottlenecks/testsets/san_francisco resnet50
'''

import sys
import os
import glob
from os.path import join

import numpy as np

from models.utils import _load_model

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from siamese_network import contrastive_loss


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

if len(sys.argv) != 4:
    print (__doc__)
    sys.exit(0)

im_dir = sys.argv[1]
output = sys.argv[2] + '.npy'
model_name = sys.argv[3]

sys.setrecursionlimit(10000)

# model = _load_model(model_name, include_top=False)
model_dir = '/tigress/dchouren/thesis/trained_models'
model = load_model(join(model_dir, model_name), custom_objects={'contrastive_loss': contrastive_loss})

labels = sorted(os.listdir(im_dir))
class_sizes = [len(os.listdir(os.path.join(im_dir, label))) for label in labels]
#ipdb.set_trace()
nb_samples = sum(class_sizes)

save_bottleneck_features(model, im_dir, (224, 224), 32, nb_samples, output)

print('Saved images to {}'.format(output))










