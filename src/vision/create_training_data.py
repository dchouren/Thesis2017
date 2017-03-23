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


def create_pos_pairs(dist, KDTree):
    return KDTree.get_pos_pairs(dist)

def create_neg_pairs(dist, KDTree):
    return KDTree.get_neg_pairs(dist)


def create_pairs(year, month, output_dir, sample_size):
    fpath_base_dir = '/tigress/dchouren/thesis/resources/paths'
    image_base_dir = '/scratch/network/dchouren/images'
    data_file = join(fpath_base_dir, year, month)
    image_dir = join(image_base_dir, year, month, month)

    with open(data_file, 'r') as inf:
        data = inf.readlines()

    # fpath files are lat, long, ...
    data_array = np.asarray([[float(x.split(',')[0]), float(x.split(',')[1]), *x.split(',')[2:]] for x in data])

    sample_data = data_array[np.random.choice(data_array.shape[0], sample_size)]
    K = _KDTree(sample_data)

    pos_dist = 0.000014  # roughly 1m
    neg_dist = 0.1838    # roughly 2000m
    pos_pairs = create_pos_pairs(pos_dist, K)

    create_pairs_helper(pos_pairs, data_array, image_dir, year, month, output_dir)

    # return pairs, labels


def create_pairs_helper(pos_pairs, all_image_data, image_dir, year, month, output_dir):

    pairs = []

    start_time = time.time()

    total_count = 0
    batch_size = 3200
    limit = 35200

    output_file = join(output_dir, year + '_' + month + '_' + str(limit) + '.h5')

    print(len(pos_pairs))

    with h5py.File(output_file, 'w') as f:

        maxshape = (None,) + (2,3,224,224)
        dset = f.create_dataset('pairs', shape=(batch_size,2,3,224,224), maxshape=maxshape, compression='gzip')
        row_count = 0

        for i, (base_data, pos_data) in enumerate(pos_pairs):
            base_filename = base_data[2].split('/')[-1]
            base_image = vutils.load_and_preprocess_image(join(image_dir, base_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)
            pos_filename = pos_data[2].split('/')[-1]
            pos_image = vutils.load_and_preprocess_image(join(image_dir, pos_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

            possible_neg_image = all_image_data[np.random.choice(all_image_data.shape[0])]
            while meter_distance(possible_neg_image[:2], base_data[:2]) < 2000:
                possible_neg_image = all_image_data[np.random.choice(
                    all_image_data.shape[0])]

            neg_filename = possible_neg_image[2].split('/')[-1]
            neg_image = vutils.load_and_preprocess_image(join(image_dir, neg_filename), dataset='imagenet', x_size=224, y_size=224, preprocess=False, rescale=True)

            if base_image is None or pos_image is None or neg_image is None:
                continue

            pos_pair = [base_image, pos_image]
            neg_pair = [base_image, neg_image]

            # ipdb.set_trace()
            pairs += [pos_pair]
            pairs += [neg_pair]

            if len(pairs) == batch_size:
                # output_file = join(output_dir, 'pairs_' + str(count).zfill(3) + '.npz')
                # print('Saving to {}'.format(output_file))
                # print(time.time() - last_time)
                # last_time = time.time()
                pairs = np.squeeze(np.asarray(pairs))
                dset[row_count:] = pairs
                row_count += pairs.shape[0]
                dset.resize(row_count + pairs.shape[0], axis=0)

                # np.savez_compressed(open(output_file, 'wb'), pairs)
                labels = []
                pairs = []

                total_count += batch_size
                print(dset.shape)


            if total_count >= limit:
                print('ending')
                dset.resize(limit, axis=0)
                print(dset.shape)
                break


        f.close()
        print('{} | Saved to {}'.format(int(time.time() - start_time), output_file))


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print (__doc__)
        sys.exit(0)

    year = sys.argv[1]
    month = sys.argv[2]
    sample_size = int(sys.argv[3])

    output = '/tigress/dchouren/thesis/resources/pairs'
    if not os.path.exists(output):
        os.makedirs(output)

    create_pairs(year, month, output, sample_size)











