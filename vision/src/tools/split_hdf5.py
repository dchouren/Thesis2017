'''
Splits large hdf5 files into smaller ones.

Usage: python split_hdf5.py [hdf5_file] [window_size]

Window size refers to the number of images you want in each split hdf5 file.
'''

import sys

import h5py

import ipdb

hdf5_file = sys.argv[1]
window_size = int(sys.argv[2])

def split_hdf5(hdf5_file):
    with h5py.File(hdf5_file, 'r') as h1:

        input = h1.get('input')
        mapping = h1.get('mapping')
        media = h1.get('media')
        location = h1.get('location')

        num_entries = len(input)
        window_start = 0
        window_end = window_start + window_size
        count = 0
        while window_start < num_entries:
            input2 = input[window_start:window_end]
            mapping2 = mapping[window_start:window_end]
            media2 = media[window_start:window_end]
            location2 = location[window_start:window_end]

            filename = hdf5_file + '_' + str(count)
            with h5py.File(filename, 'w') as h2:
                d1 = h2.create_dataset('input', data=input2)
                d2 = h2.create_dataset('mapping', data=mapping2)
                d3 = h2.create_dataset('media', data=media2)
                d4 = h2.create_dataset('location', data=location2)

            window_start = window_end
            window_end += window_size
            count += 1

            print('Wrote {}'.format(filename))

split_hdf5(hdf5_file)