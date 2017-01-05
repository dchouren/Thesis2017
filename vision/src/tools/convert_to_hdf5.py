'''
usage: python convert_to_hdf5.py [task] [dataset] [im_set]
example: python src/convert_to_hdf5.py story 60763_10022 all

Convert an image dir to hdf5 file
'''
#!/usr/bin/env python

import numpy as np
import h5py
from scipy.misc import imread, imresize
import ipdb
import sys
from os.path import join
from _utils import get_fpath_filename, mkdir_p

task = sys.argv[1]
dataset = sys.argv[2]
im_dir = sys.argv[3]

if len(sys.argv) != 4:
    print (__doc__)
    sys.exit(0)

formatted_paths_file = join('resources/fpaths', task, dataset, im_set)
hdf5_dir = join('/media/storage/scratch/dchouren_resources/hdf5', task, dataset)

try:
    mkdir_p(im_dir)
    mkdir_p(hdf5_dir)
except:
    print("error making im dir or hdf5 dir")
hdf5_file = join(hdf5_dir, im_set)


size = (100,100)


def write_to_hdf5(input_images, mapping_urls, mediaids, locationids, filename):
    input = np.asarray(input_images)
    mapping = np.asarray(mapping_urls)
    media = np.asarray(mediaids)
    location = np.asarray(locationids)

    # Create the HDF5 file
    with h5py.File(filename, 'w') as h1:
        d1 = h1.create_dataset('input', data=input)
        d2 = h1.create_dataset('mapping', data=mapping)
        d3 = h1.create_dataset('media', data=media)
        d4 = h1.create_dataset('location', data=location)

        # Close the file
        h1.close()
        print('Wrote to {}'.format(filename))


input_images = []
mapping_urls = []
mediaids = []
locationids = []
counter = 1

with open(formatted_paths_file, 'r') as inf:
    for line in inf.readlines():
        tokens = line.split(',')
        fpath = tokens[0]
        mediaid = tokens[1]
        locationid = tokens[2]

        im_file = get_fpath_filename(fpath)
        # ipdb.set_trace()
        try:
            im = imread(join(im_dir, im_file.strip()), mode='RGB').astype('float64')
            im = imresize(im, size)
            im = im / 255.
        except:
            print('Failure: Could not read image: ' + im_file)
            continue
        # move RGB to first axis
        if len(im.shape) < 3:
            print('Failure: Less than 3 axes: ' + im_file)
            continue
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        input_images.append(im)
        mapping_urls.append(np.array([(fpath+'\n').encode('utf-8')]))
        mediaids.append(np.array([(mediaid+'\n').encode('utf-8')]))
        locationids.append(np.array([(locationid+'\n').encode('utf-8')]))

    # ipdb.set_trace()

    write_to_hdf5(input_images, mapping_urls, mediaids, locationids, hdf5_file)


# y = h5py.File('test.h5', 'r')
#
# ipdb.set_trace()