'''
Create image training triplets.

Usage: python create_training_data.py year month output_file
Example: python src/vision/create_training_data.py 2014 01 resources/pairs/2014
'''

import sys
from os.path import join
import os
import time

import numpy as np
import h5py

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
    image_base_dir = '/scratch/network/dchouren/images'
    data_file = join(fpath_base_dir, data_file)
    # image_dir = join(image_base_dir, year, month, month)

    with open(data_file, 'r') as inf:
        data = inf.readlines()

    # fpath files are lat, long, ...
    data_array = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data if not x.split(',')[2].endswith('zz=1')])

    sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    K = _KDTree(sample_data)

    pos_dist = 0.000014  # roughly 1m
    pos_dist = 0.0002 # roughly 10m
    neg_dist = 0.01838    # roughly 2000m
    possible_neg_dist = 0.001 # roughly 111 meters

    pos_pairs = create_pos_pairs(pos_dist, K)

    diff_user = np.array([x for x in pos_pairs if x[0][5] != x[1][5]])
    same_user = np.array([x for x in pos_pairs if x[0][5] == x[1][5]][:diff_user.shape[0]])


    load_pairs(same_user, diff_user, image_base_dir, output_file)


    ipdb.set_trace()

    # possible_neg_pairs = create_pos_pairs(possible_neg_dist, K)

    # fulfilled = create_pairs_helper(pos_pairs, data_array, image_dir, year, month, output_dir)

    # count = 1
    # while not fulfilled:
    #     sample_size *= 2
    #     sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    #     K = _KDTree(sample_data)

    #     pos_dist = 0.000014  # roughly 1m
    #     neg_dist = 0.1838    # roughly 2000m
    #     pos_pairs = create_pos_pairs(pos_dist, K)
    #     # possible_neg_pairs = create_pos_pairs(possible_neg_dist, K)
    #     if len(pos_pairs) < 35200:
    #         fulfilled = False
    #         continue

    #     fulfilled = create_pairs_helper(pos_pairs, data_array, image_dir, year, month, output_dir)
    #     count += 1

    #     if count > 5:
    #         print('Failed to create enough pairs')
    #         return

    # return pairs, labels

def load_pairs(same_pairs, diff_pairs, image_dir, output_file):

    pairs = []
    labels = []

    total_count = 0
    batch_size = 3200

    start_time = time.time()
    with h5py.File(output_file, 'w') as f:
        maxshape = (None,) + (2,3,224,224)
        dset = f.create_dataset('pairs', shape=(batch_size,2,3,224,224), maxshape=maxshape)

        row_count = 0

        for i, (base_data, pair_data) in enumerate(zip(same_pairs, diff_pairs)):
            base_filename = base_data[2].split('/')[-1]
            year, month = base_data[3].split('-')[0], base_data[3].split('-')[1]
            base_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, base_filename), dataset='flickr', x_size=224, y_size=224, preprocess=True, rescale=True)
            if base_image is None:
                continue

            pair_filename = pair_data[2].split('/')[-1]
            year, month = pair_data[3].split('-')[0], pair_data[3].split('-')[1]
            pair_image = vutils.load_and_preprocess_image(join(image_dir, year, month, month, pair_filename), dataset='flickr', x_size=224, y_size=224, preprocess=True, rescale=True)
            if pair_image is None:
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
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)

                labels = []
                pairs = []

                total_count += batch_size

    f.close()
    print('{} | Saved to {}'.format(int(time.time() - start_time), output_file))





def create_pairs_helper(pos_pairs, all_image_data, image_dir, year, month, output_dir):

    pairs = []

    start_time = time.time()

    total_count = 0
    batch_size = 3200
    limit = 35200

    output_file = join(output_dir, year + '_' + month + '_' + str(limit) + '_new.h5')

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
            print('finding negative')
            i = 0
            while distance < 10 or distance > 100:
                print(distance)
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

            print('{}: {}'.format(len(pairs), int(time.time() - start_time)))

            if len(pairs) == batch_size:
                pairs = np.squeeze(np.asarray(pairs))
                dset[row_count:] = pairs
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)

                # np.savez_compressed(open(output_file, 'wb'), pairs)
                labels = []
                pairs = []

                total_count += batch_size
                # print(dset.shape)

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
            return False


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











