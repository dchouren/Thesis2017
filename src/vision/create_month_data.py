'''
Create image training triplets.

Usage: python create_training_data.py year month output_file
Example: python src/vision/create_training_data.py 2014 01 resources/pairs/2014
'''

import sys
from os.path import join
import os
import time
from datetime import datetime

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


def create_pairs(year, month, sample_size, limit):
    fpath_base_dir = '/tigress/dchouren/thesis/resources/paths'
    data_file = join(fpath_base_dir, year, month)

    image_dir = join('/scratch/network/dchouren/images/', year, month, month)
    output_dir = '/tigress/dchouren/thesis/resources/pairs'
    output_file = join(output_dir, year + '_' + month + '.h5')

    with open(data_file, 'r') as inf:
        data = inf.readlines()

    # fpath files are lat, long, ...
    data_array = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data if not x.split(',')[2].endswith('zz=1')])

    sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    K = _KDTree(sample_data)

    pos_dist = 0.000014  # roughly 1m
    pos_dist = 0.0001 # roughly 10m
    neg_dist = 0.01838    # roughly 2000m
    possible_neg_dist = 0.001 # roughly 111 meters

    pos_pairs = create_pos_pairs(pos_dist, K)
    same_user = np.array([x for x in pos_pairs if x[0][5] == x[1][5] and '66430340@N07' not in x[0][5]])

    fulfilled = create_pairs_helper(same_user, data_array, image_dir, year, month, limit)

    count = 1
    while not fulfilled:
        sample_size *= 2
        sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
        K = _KDTree(sample_data)

        pos_dist = 0.000014  # roughly 1m
        neg_dist = 0.1838    # roughly 2000m
        pos_pairs = create_pos_pairs(pos_dist, K)
        same_user = np.array([x for x in pos_pairs if x[0][5] == x[1][5] and '66430340@N07' not in x[0][5]])
        # possible_neg_pairs = create_pos_pairs(possible_neg_dist, K)
        if len(same_user) < 1600:
            fulfilled = False
            continue

        fulfilled = create_pairs_helper(same_user, data_array, image_dir, year, month, limit)
        count += 1

        if count > 5:
            print('Failed to create enough pairs')
            return



def create_pairs_helper(pos_pairs, all_image_data, image_dir, year, month, limit):

    pairs = []

    start_time = time.time()

    total_count = 0
    batch_size = 1600
    # limit = 35200

    output_dir = '/tigress/dchouren/thesis/resources/pairs'
    output_file = join(output_dir, year + '_' + month + '_' + str(limit) + '.h5')

    print(len(pos_pairs))

    with h5py.File(output_file, 'w') as f:

        maxshape = (None,) + (2,3,224,224)
        dset = f.create_dataset('pairs', shape=(batch_size,2,3,224,224), maxshape=maxshape, compression='gzip')
        row_count = 0

        for i, (base_data, pos_data) in enumerate(pos_pairs):
            base_filename = base_data[2].split('/')[-1]
            if base_filename.endswith('zz=1'):
                continue
            base_image = vutils.load_and_preprocess_image(join(image_dir, base_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            if base_image is None:
                continue

            pos_filename = pos_data[2].split('/')[-1]
            if pos_filename.endswith('zz=1'):
                continue
            pos_image = vutils.load_and_preprocess_image(join(image_dir, pos_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            if pos_image is None:
                continue

            possible_neg_image = all_image_data[np.random.choice(all_image_data.shape[0])]

            # ipdb.set_trace()
            distance = meter_distance(possible_neg_image[:2], base_data[:2])
            # print('finding negative')
            i = 0
            while distance < 2000:
                # print(distance)
                possible_neg_image = all_image_data[np.random.choice(
                    all_image_data.shape[0])]
                distance = meter_distance(possible_neg_image[:2], base_data[:2])
                i += 1

            neg_filename = possible_neg_image[2].split('/')[-1]
            if neg_filename.endswith('zz=1'):
                continue
            neg_image = vutils.load_and_preprocess_image(join(image_dir, neg_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            if neg_image is None:
                continue


            pos_pair = [base_image, pos_image]
            neg_pair = [base_image, neg_image]

            # ipdb.set_trace()
            pairs += [pos_pair]
            pairs += [neg_pair]

            # print('{}: {}'.format(len(pairs), int(time.time() - start_time)))

            if len(pairs) == batch_size:
                # print('here')
                # ipdb.set_trace()
                pairs = np.squeeze(np.asarray(pairs))
                # print('squeezed pairs')
                dset[row_count:] = pairs
                # print('assigned dset')
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)
                # print('resized dset')

                # np.savez_compressed(open(output_file, 'wb'), pairs)
                labels = []
                pairs = []

                total_count += batch_size
                print(dset.shape)

            if total_count >= limit:
                # print('ending')
                dset.resize(limit, axis=0)
                print(dset.shape)
                break

        if dset.shape[0] == limit:
            f.close()
            print('{} | Saved to {}'.format(int(time.time() - start_time), output_file))
            return True
        else:
            del dset
            del f
            print('False')
            return False


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print (__doc__)
        sys.exit(0)

    sys.setrecursionlimit(10000)

    year = sys.argv[1]
    month = sys.argv[2]
    sample_size = int(sys.argv[3])
    limit = int(sys.argv[4])

    output = '/tigress/dchouren/thesis/resources/pairs'
    if not os.path.exists(output):
        os.makedirs(output)

    # output_file = join(output, output_file)

    create_pairs(year, month, sample_size, limit)











