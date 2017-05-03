'''
Create image training triplets.

Usage: python create_training_data.py year month output_file
Example: python src/vision/create_training_data.py 2014 01 resources/pairs/2014
'''

import sys
from os.path import join
import os
import time
from datetime import datetime, timedelta

import numpy as np
import h5py
import pickle

from _KDTree import _KDTree
import vision_utils as vutils
from gen_utils import meter_distance

import ipdb


def create_pos_pairs(dist, KDTree):
    return np.array(KDTree.get_pos_pairs(dist))

def create_neg_pairs(dist, KDTree):
    return KDTree.get_neg_pairs(dist)


def pos_pairs_same_user(pos_pairs):
    same_user = [x for x in pos_pairs if x[0][5] == x[1][5]]


def create_pairs(data_file, output_file, sample_size):
    fpath_base_dir = '/tigress/dchouren/thesis/resources/paths'
    image_base_dir = '/scratch/network/dchouren/images/'
    data_file = join(fpath_base_dir, data_file)
    # image_dir = join(image_base_dir, year, month, month)

    with open(data_file, 'r') as inf:
        data = inf.readlines()

    # fpath files are lat, long, ...
    data_array = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data if not x.split(',')[2].endswith('zz=1')])

    sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    K = _KDTree(sample_data)

    # pos_dist = 0.000014  # roughly 1m
    pos_dist = 0.0001 # roughly 10m
    # pos_dist = 0.0003 # roughly 30m
    neg_dist = 0.01838    # roughly 2000m
    possible_neg_dist = 0.001 # roughly 111 meters

    pos_pairs = create_pos_pairs(pos_dist, K)

    diff_user = np.array([x for x in pos_pairs if x[0][5] != x[1][5]])
    same_user = np.array([x for x in pos_pairs if x[0][5] == x[1][5] and '66430340@N07' not in x[0][5]])[:diff_user.shape[0]]
    # same_user = np.array([x for x in pos_pairs if x[0][5] == x[1][5] and '66430340@N07' not in x[0][5] and abs(datetime.strptime(x[0][3],'%Y-%m-%d %H:%M:%S') - datetime.strptime(x[1][3], '%Y-%m-%d %H:%M:%S')) < timedelta(seconds=7200)][:diff_user.shape[0]])

    diff_user = diff_user[:same_user.shape[0]]

    # ipdb.set_trace()

    # same_dates = [(datetime.strptime(x[0][3],'%Y-%m-%d %H:%M:%S'),datetime.strptime(x[1][3], '%Y-%m-%d %H:%M:%S')) for x in same_user]
    # diff_dates = [(datetime.strptime(x[0][3],'%Y-%m-%d %H:%M:%S'),datetime.strptime(x[1][3], '%Y-%m-%d %H:%M:%S')) for x in diff_user]

    # same_diffs = [abs(x[0] - x[1]) for x in same_dates]
    # diff_diffs = [abs(x[0] - x[1]) for x in diff_dates]

    # with open('/tigress/dchouren/thesis/same_diffs.pickle', 'wb') as outf:
    #     pickle.dump(same_diffs, outf)
    # with open('/tigress/dchouren/thesis/diff_diffs.pickle', 'wb') as outf:
    #     pickle.dump(diff_diffs, outf)   

    # ipdb.set_trace()


    # load_middlebury(diff_user, image_base_dir)
    load_pairs(same_user, diff_user, image_base_dir, output_file)


    # possible_neg_pairs = create_pos_pairs(possible_neg_dist, K)


def load_pairs(same_pairs, diff_pairs, image_dir, output_file):

    pairs = []
    labels = []

    total_count = 0
    batch_size = 1600

    start_time = time.time()
    with h5py.File(output_file, 'w') as f:
        maxshape = (None,) + (2,3,224,224)
        dset = f.create_dataset('pairs', shape=(batch_size,2,3,224,224), maxshape=maxshape)
        labels_dset = f.create_dataset('labels', shape=(batch_size,1), maxshape=(None, 1))

        row_count = 0

        # ipdb.set_trace()
        data = np.empty((same_pairs.shape[0]*2, same_pairs.shape[1]), dtype=same_pairs.dtype)
        # ipdb.set_trace()
        data[0::2] = same_pairs
        data[1::2] = diff_pairs
        # ipdb.set_trace()
        for i, (base_data, pair_data) in enumerate(data):
            # ipdb.set_trace()
            base_filename = base_data[2].split('/')[-1]
            year, month = base_data[3].split('-')[0], base_data[3].split('-')[1]
            if base_filename.endswith('zz=1'):
                continue
            base_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, base_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            if base_image is None:
                # print(i)
                continue

            pair_filename = pair_data[2].split('/')[-1]
            year, month = pair_data[3].split('-')[0], pair_data[3].split('-')[1]
            if pair_filename.endswith('zz=1'):
                continue
            pair_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, pair_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            if pair_image is None:
                # print(i)
                continue

            pair = [base_image, pair_image]
            if i % 2 == 0:
                label = 1
            else:
                label = 0

            pairs += [pair]
            labels += [label]

            if len(pairs) == batch_size:
                pairs = np.squeeze(np.asarray(pairs))
                dset[row_count:] = pairs
                labels = np.array(labels)
                labels_dset[row_count:] = np.reshape(labels, (labels.shape[0], 1))
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)
                labels_dset.resize(row_count + pairs.shape[0], axis=0)

                labels = []
                pairs = []

                total_count += batch_size

        if dset.shape[0] >= data.shape[0]:
            dset.resize(data.shape[0], axis=0)
    print('{} | Saved to {}'.format(int(time.time() - start_time), output_file))



def load_middlebury(match_pairs, image_dir):

    middlebury_pairs = []
    middlebury_dir = '/tigress/dchouren/thesis/resources/stereo/images'
    filenames = sorted(os.listdir(middlebury_dir))
    quads = list(zip(filenames[::4], filenames[1::4], filenames[2::4], filenames[3::4]))

    all_pairs = []
    stereo_pairs = []
    for quad in quads:
        pairs = [(quad[0], quad[1]), (quad[0], quad[2]), (quad[0], quad[3])]
        all_pairs.extend(pairs)
        stereo_pairs.extend([(quad[0], quad[1])])

    all_middlebury_images = []
    all_stereo_images = []
    for base_file, pos_file in all_pairs:
        base_image = vutils.load_and_preprocess_image(join(middlebury_dir, base_file), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
        pos_image = vutils.load_and_preprocess_image(join(middlebury_dir, pos_file), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

        all_middlebury_images.append([base_image, pos_image])

    for base_file, pos_file in stereo_pairs:
        base_image = vutils.load_and_preprocess_image(join(middlebury_dir, base_file), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
        pos_image = vutils.load_and_preprocess_image(join(middlebury_dir, pos_file), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

        all_stereo_images.append([base_image, pos_image])

    all_middlebury_images = np.squeeze(np.array(all_middlebury_images))
    all_stereo_images = np.squeeze(np.array(all_stereo_images))

    # ipdb.set_trace()

    pairs = []
    labels = []

    total_count = 0
    batch_size = 1600

    # ipdb.set_trace()
    print(match_pairs.shape[0])

    start_time = time.time()
    with h5py.File(output_file, 'w') as f:
        maxshape = (None,) + (2,3,224,224)
        dset = f.create_dataset('pairs', shape=(batch_size,2,3,224,224), maxshape=maxshape)
        labels_dset = f.create_dataset('labels', shape=(batch_size,1), maxshape=(None, 1))

        row_count = 0

        for i, (base_data, pair_data) in enumerate(np.vstack(zip(match_pairs, match_pairs))):
            if i % 2 == 0:
                pair = all_stereo_images[np.random.choice(all_stereo_images.shape[0])]
                label = 1
            else:
                base_filename = base_data[2].split('/')[-1]
                year, month = base_data[3].split('-')[0], base_data[3].split('-')[1]
                base_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, base_filename), dataset='flickr', x_size=224, y_size=224, preprocess=False, rescale=True)
                if base_image is None:
                    continue

                pair_filename = pair_data[2].split('/')[-1]
                year, month = pair_data[3].split('-')[0], pair_data[3].split('-')[1]
                pair_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, pair_filename), dataset='flickr', x_size=224, y_size=224, preprocess=False, rescale=True)
                if pair_image is None:
                    continue

                pair = np.squeeze(np.array([base_image, pair_image]))
                label = 0

            pairs += [pair]
            labels += [label]

            if len(pairs) == batch_size:
                # print('here')
                # pairs = np.squeeze(np.asarray(pairs))
                # ipdb.set_trace()
                pairs = np.asarray(pairs)
                dset[row_count:] = pairs
                labels = np.array(labels)
                labels_dset[row_count:] = np.reshape(labels, (labels.shape[0], 1))
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)
                labels_dset.resize(row_count + pairs.shape[0], axis=0)

                labels = []
                pairs = []

                total_count += batch_size

    print('{} | Saved to {}'.format(int(time.time() - start_time), output_file))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print (__doc__)
        sys.exit(0)

    sys.setrecursionlimit(10000)

    data_file = sys.argv[1]
    output_file = sys.argv[2]
    sample_size = int(sys.argv[3])

    output = '/tigress/dchouren/thesis/resources/pairs'
    if not os.path.exists(output):
        os.makedirs(output)

    output_file = join(output, output_file)

    create_pairs(data_file, output_file, sample_size)



# ./src/generate_slurm.sh 144:00:00 62GB "python /tigress/dchouren/thesis/src/vision/siamese_network.py resnet50 nadam 50 f2013-5_1m_user.h5" f2013-5_1m_user.h5 true dchouren@princeton.edu /tigress/dchouren/thesis







