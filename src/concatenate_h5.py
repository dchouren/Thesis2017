import sys
from os.path import join
import time
from copy import copy

import h5py
import numpy as np

import ipdb

pairs_dir = '/tigress/dchouren/thesis/resources/pairs'
filenames = open(join(pairs_dir, 'train_files.txt'), 'r').readlines()
filenames = [filename[2:].strip() for filename in filenames]

output_file = sys.argv[1]

start_time = time.time()

output_file = h5py.File(join(pairs_dir, output_file), 'w')

total_rows = 0
last_index = 0

indexes = []
for i, filename in enumerate(filenames):
    print(filename)
    with h5py.File(join(pairs_dir, filename), 'r') as h5:
        pairs = h5['pairs']
        # print(pairs.shape)

        indexes += [pairs.shape[0]]

print(indexes)
indexes = np.cumsum(indexes)
# ipdb.set_trace()

pairs_dataset = output_file.create_dataset('pairs', shape=(indexes[-1], 2, 3, 224, 224), maxshape=(indexes[-1], 2, 3, 224, 224))

means = []

for end_index, filename in zip(indexes, filenames):
    f = h5py.File(join(pairs_dir, filename), 'r')
    pairs = copy(f['pairs'])

    # ipdb.set_trace()
    print(filename)
    # pairs_dataset.resize(total_rows, axis=0)
    pairs_dataset[total_rows:end_index] = pairs
    total_rows = end_index

    mean = np.mean(pairs, axis=(0,1,3,4))
    means += [mean]

    f.close()
    print(int(time.time() - start_time))
        # print(pairs.shape)
        # print(pairs[0])
        # print(pairs[-1])
        # print(total_rows)

        # if i == 0:
        #     pairs_dataset = output_file.create_dataset('pairs', shape=(pairs.shape[0], 2, 3, 224, 224), maxshape=(None, 2, 3, 224, 224), compression='gzip')
        #     pairs_dataset[total_rows:] = pairs
        #     total_rows += pairs.shape[0]
        #     pairs_dataset.resize(total_rows, axis=0)
        #     # last_index = total_rows
        # else:
        #     pairs_dataset[total_rows:] = pairs
        #     total_rows += pairs.shape[0]
        #     pairs_dataset.resize(total_rows, axis=0)

# pairs_dataset.resize(total_rows-pairs.shape[0], axis=0)

output_file.close()
np.save('/tigress/dchouren/thesis/test_means.npy', means)

print(int(time.time() - start_time))

sys.exit(0)












