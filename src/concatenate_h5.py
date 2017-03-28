import sys
from os.path import join

import h5py

pairs_dir = '/tigress/dchouren/thesis/resources/pairs'
filenames = open(join(pairs_dir, 'train_files.txt'), 'r').readlines()[:2]

output_file = sys.argv[1]

with h5py.File(join(pairs_dir, output_file), 'w') as output_file:

    total_rows = 0
    last_index = 0
    for i, filename in enumerate(filenames):
        filename = filename[2:].strip()
        print(filename)
        h5 = h5py.File(join(pairs_dir, filename), 'r')
        pairs = h5['pairs']
        print(pairs.shape)

        # total_rows = total_rows + pairs.shape[0]
        print(total_rows)

        if i == 0:
            pairs_dataset = output_file.create_dataset('pairs', shape=(pairs.shape[0], 2, 3, 224, 224), maxshape=(None, 2, 3, 224, 224), compression='gzip')
            pairs_dataset[total_rows:] = pairs
            total_rows += pairs.shape[0]
            pairs_dataset.resize(total_rows, axis=0)
            # last_index = total_rows
        else:
            pairs_dataset[total_rows:] = pairs
            total_rows += pairs.shape[0]
            pairs_dataset.resize(total_rows, axis=0)

    pairs_dataset.resize(total_rows-pairs.shape[0], axis=0)