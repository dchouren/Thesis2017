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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.stock_resnet50 import SResNet50


from models.utils import _load_model
from siamese_network import contrastive_loss
from format_h5 import load_path_map


import ipdb


def save_bottleneck_features(model, year, month, output_path,img_size=(224,224), batch_size=32):

    start_time = time.time() 

    datagen = ImageDataGenerator(rescale=1. / 255)

    directory = join('/scratch/network/dchouren/images', year, month)

    labels = sorted(os.listdir(directory))
    class_sizes = [len(os.listdir(os.path.join(directory, label))) for label in labels]
    nb_samples = sum(class_sizes)
    # nb_samples = 96

    # ipdb.set_trace()

    print('Running {}'.format(directory))
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        shuffle=False)

    filenames = np.array([x.split('/')[-1].encode('utf8') for x in generator.filenames])
    bottleneck_features = model.predict_generator(generator, nb_samples/batch_size)

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


def _generator(filename, batch_size=32, index=0, augment=False):
    f = h5py.File(filename, 'r')
    pairs = f['pairs']
    idg = ImageDataGenerator(rotation_range=20, horizontal_flip=True, vertical_flip=True, zoom_range=0.05)
    while 1:
        data = pairs[index:index + int(batch_size/2)]
        if augment:
            data = np.array([[idg.random_transform(x[0]), idg.random_transform(x[1])] for x in data])

        left = data[:,0]
        right = data[:,1]
        gen_batch = np.empty((left.shape[0] + right.shape[0], *left.shape[1:]), dtype=left.dtype)
        gen_batch[0::2] = left
        gen_batch[1::2] = right
        yield gen_batch
        index += batch_size
        index = index % len(pairs)
        # print(index)

    f.close()


def extract_bottlenecks(model, pairs_file, output):

    batch_size = 32
    with h5py.File(pairs_file) as inf:
        total_images = inf['pairs'].shape[0] * 2
    generator = _generator(pairs_file)

    print('Predicting')
    bottlenecks = model.predict_generator(generator, steps=int(total_images/batch_size))

    pos_distances, pos_mean, pos_median, neg_distances, neg_mean, neg_median = compute_distance_metrics(bottlenecks, 'a')
    np.save(output + '_posdist', pos_distances)
    np.save(output + '_negdist', neg_distances)


    print('Pos Mean: {}'.format(pos_mean))
    print('Pos Median: {}'.format(pos_median))
    print('Neg Mean: {}'.format(neg_mean))
    print('Neg Median: {}'.format(neg_median))
    np.save(output, bottlenecks)
    print('Saved to {}'.format(output))


def compute_distance_metrics(bottlenecks, filebase):
    pos_distances = np.array([np.linalg.norm(left - right) for left, right in zip(bottlenecks[0::4], bottlenecks[1::4])])
    neg_distances = np.array([np.linalg.norm(left - right) for left, right in zip(bottlenecks[2::4], bottlenecks[3::4])])

    x = pos_distances
    y = neg_distances

    xweights = 100 * np.ones_like(x) / x.size
    yweights = 100 * np.ones_like(y) / y.size
    bins = 20
    fig, ax = plt.subplots()
    ax.hist(x, bins, color='lightblue', alpha=0.5, label='Positive', normed=True)
    ax.hist(y, bins, color='salmon', alpha=0.5, label='Negative', normed=True)

    # ax.set(ylabel='% of Distances in Bin')
    ax.margins(0.05)
    ax.set_ylim(bottom=0)
    legend = ax.legend(loc='upper right')
    plt.savefig('/tigress/dchouren/thesis/plots/distances/' + filebase)
    plt.gcf()

    inversions = sum(pos_distances > neg_distances)
    inversion_pct = inversions / len(pos_distances)
    print('Inversions: {}'.format(inversion_pct))

    pos_mean = np.mean(pos_distances)
    pos_median = np.median(pos_distances)
    neg_mean = np.mean(neg_distances)
    neg_median = np.median(neg_distances)


    print('Pos Mean: {}'.format(pos_mean))
    print('Pos Median: {}'.format(pos_median))
    print('Neg Mean: {}'.format(neg_mean))
    print('Neg Median: {}'.format(neg_median))

    return pos_distances, pos_mean, pos_median, neg_distances, neg_mean, neg_median



def main():
    # if len(sys.argv) != 4:
    #     print (__doc__)
    #     sys.exit(0)

    model_name = sys.argv[1]
    year = sys.argv[2]
    month = sys.argv[3]

    # output = join('/tigress/dchouren/thesis/resources/bottlenecks', model_name, year + '_' + month + '.h5')
    output = '/tigress/dchouren/thesis/bottleneck2.h5'

    sys.setrecursionlimit(10000)

    model_dir = '/tigress/dchouren/thesis/trained_models/'

    model = SResNet50(include_top=False, weights='imagenet')

    # model = load_model(join(model_dir, model_name), custom_objects={'contrastive_loss': contrastive_loss})
    model.load_weights(join(model_dir, model_name), by_name=True)

    # ipdb.set_trace()

    save_bottleneck_features(model, year, month, output, (224, 224), 32)




    # pairs = sys.argv[1]

    # pairs_dir = '/tigress/dchouren/thesis/resources/pairs/'
    # pairs_file = join(pairs_dir, pairs)
    # filebase = pairs.split('.h5')[0]

    # output_dir = '/tigress/dchouren/thesis/resources/bottlenecks/'
    # output_file = join(output_dir, filebase + '.npy')


    # # extract_bottlenecks(model, pairs_file, output_file)

    # filebases = ["2015_01_32000.h5", "2014_02_32000.h5", "2015_02_32000.h5", "2015_03_32000.h5", "middlebury_diffuser.h5", "2013-5_10m_tl.h5", "2014_03_32000.h5", "2015_04_32000.h5", "2013-5_10m_user.h5", "2014_04_32000.h5", "2015_05_32000.h5", "2014_05_32000.h5", "2015_06_32000.h5", "middlebury_sameuser.h5", "2014_06_32000.h5", "2015_07_32000.h5", "new_2013_2014_2015_all.h5", "2013-5_1m_tl.h5", "2014_07_32000.h5", "2015_08_32000.h5", "new_2015_all.h5", "2013-5_1m_user.h5", "2014_08_32000.h5", "2015_09_32000.h5", "2013-5_30m_user_2h.h5", "2014_09_32000.h5", "2015_10_32000.h5", "stereo_diffuser.h5", "2014_01_32000.h5", "2014_10_32000.h5", "2015_11_32000.h5", "stereo_sameuser.h5", "2014_11_32000.h5", "2015_12_32000.h5", "2014_12_32000.h5"]
    # # filebases = ['a2013-5_1m_user.h5']

    # for filebase in sorted(filebases):
    #     filebase = filebase.split('.')[0]
    #     bottleneck_file = join(output_dir, filebase + '.npy')
    #     try:
    #         bottlenecks = np.load(bottleneck_file)
    #         print(filebase)
    #         compute_distance_metrics(bottlenecks, filebase)
    #         print()
    #     except:
    #         pass


    # model_file = sys.argv[1]






if __name__ == '__main__':
    sys.exit(main())






