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
from keras.applications.resnet50 import ResNet50


from models.utils import _load_model
from siamese_network import contrastive_loss, _generator
from format_h5 import load_path_map


import ipdb


def save_bottleneck_features(model, year, month, output_path,img_size=(224,224), batch_size=32):

    start_time = time.time() 

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

    for f in filenames:
        if f not in path_map:
            path_map[f] = [''] * 8

    dates = [path_map[x][1].encode("ascii", "ignore") for x in filenames]
    lats = [path_map[x][2].encode("ascii", "ignore") for x in filenames]
    lons = [path_map[x][3].encode("ascii", "ignore") for x in filenames]
    zooms = [path_map[x][4].encode("ascii", "ignore") for x in filenames]
    users = [path_map[x][5].encode("ascii", "ignore") for x in filenames]
    titles = [path_map[x][6].encode("ascii", "ignore") for x in filenames]
    descriptions = [path_map[x][7].encode("ascii", "ignore") for x in filenames]

    f = h5py.File(output_path, 'w')
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

    print('{}s | Saved images to {}'.format(int(time.time() - start_time), output_path))



def extract_bottlenecks(model, pairs_file, output):

    batch_size = 32
    with h5py.File(pairs_file) as inf:
        total_pairs = inf['pairs'].shape[0]
    generator = _generator(pairs_file)
    bottlenecks = model.predict_generator(generator, steps=int(total_pairs/batch_size))
    np.save(output, bottlenecks)


def main():
    # if len(sys.argv) != 4:
    #     print (__doc__)
    #     sys.exit(0)

    # model_name = sys.argv[1]
    # year = sys.argv[2]
    # month = sys.argv[3]

    # output = join('/tigress/dchouren/thesis/resources/bottlenecks', model_name, year + '_' + month + '.h5')

    # sys.setrecursionlimit(10000)

    # model_dir = '/tigress/dchouren/thesis/trained_models/base_cnn'
    # model = load_model(join(model_dir, model_name))

    # save_bottleneck_features(model, year, month, output, (224, 224), 32)

    model = ResNet50(weights='imagenet')
    pairs_file = sys.argv[1]
    output_file = sys.argv[2]

    extract_bottlenecks(model, pairs_file, output_file)




if __name__ == '__main__':
    sys.exit(main())






