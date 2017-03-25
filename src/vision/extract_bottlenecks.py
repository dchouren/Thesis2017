'''
Run images through a topless base model and save features for future predicting/training. You need to put the image directory you want to save in a wrapper directory. Eg, if you want to save san_francisco images, you need to hide your images in $IMAGES/testsets/wrapper/san_francisco, where san_francisco is the only directory in wrapper/.

usage: python extract_bottlenecks.py im_dir output model_name
example: python src/vision/extract_bottlenecks.py $IMAGES/testsets/wrapper $RESOURCES/bottlenecks/testsets/san_francisco resnet50
'''

import sys
import os
import glob
from os.path import join
import time

import numpy as np
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

from models.utils import _load_model
from siamese_network import contrastive_loss
from format_h5 import load_path_map


import ipdb


def save_bottleneck_features(model, year, month, img_size, batch_size, output_path):
    datagen = ImageDataGenerator(rescale=1. / 255)

    directory = join('/scratch/network/dchouren/images', year, month)

    labels = sorted(os.listdir(directory))
    class_sizes = [len(os.listdir(os.path.join(directory, label))) for label in labels]
    nb_samples = sum(class_sizes)

    print('Running {}'.format(directory))
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False)

    filenames = np.array([x.split('/')[-1].encode('utf8') for x in generator.filenames])
    bottleneck_features = model.predict_generator(generator, nb_samples)

    path_map = load_path_map(join('/tigress/dchouren/thesis/resources/paths/', year, month.zfill(2)))
    dates = [path_map[x][1].encode("ascii", "ignore") for x in filenames]
    lats = [path_map[x][2].encode("ascii", "ignore") for x in filenames]
    lons = [path_map[x][3].encode("ascii", "ignore") for x in filenames]
    zooms = [path_map[x][4].encode("ascii", "ignore") for x in filenames]
    users = [path_map[x][5].encode("ascii", "ignore") for x in filenames]
    titles = [path_map[x][6].encode("ascii", "ignore") for x in filenames]
    descriptions = [path_map[x][7].encode("ascii", "ignore") for x in filenames]

    f = h5py.File(output_path + '.h5', 'w')
    f.create_dataset('bottlenecks', data=bottleneck_features)
    f.create_dataset('filenames', data=filenames)
    f.create_dataset('dates', data=dates)
    f.create_dataset('lats', data=lats)
    f.create_dataset('lons', data=lons)
    f.create_dataset('zooms', data=zooms)
    f.create_dataset('users', data=users)
    f.create_dataset('titles', data=titles)
    f.create_dataset('descriptions', data=descriptions)
    f.close()
    # np.save(open(output_path, 'wb'), bottleneck_features)


def main():
    if len(sys.argv) != 4:
        print (__doc__)
        sys.exit(0)

    year = sys.argv[1]
    month = sys.argv[2]
    model_name = sys.argv[3]

    output = join('/tigress/dchouren/thesis/resources/bottlenecks', model_name, year + '_' + month + '.h5')

    sys.setrecursionlimit(10000)

    start_time = time.time()

    # model = _load_model(model_name, include_top=False)
    model_dir = '/tigress/dchouren/thesis/trained_models'
    model = load_model(join(model_dir, model_name))
    # model = load_model(join(model_dir, model_name), custom_objects={'contrastive_loss': contrastive_loss})
    # ipdb.set_trace()

    save_bottleneck_features(model, year, month, (224, 224), 32, output)

    print('{}s | Saved images to {}'.format(int(time.time() - start_time), output))



if __name__ == '__main__':
    sys.exit(main())






