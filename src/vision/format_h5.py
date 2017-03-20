import sys
import glob
import h5py
import os

import numpy as np

from os.path import join
import time

import ipdb

bottleneck_dir = sys.argv[1]
output_dir = sys.argv[2]
image_dir = sys.argv[3]
paths_dir = sys.argv[4]

bottlenecks = glob.glob(join(bottleneck_dir, '*.npy'))

months = ['01','02','03','04','05','06','07','08','09','10','11','12']

def load_path_map(path_file):
    path_map = {}
    with open(path_file, 'r') as inf:
        for line in inf:
            tokens = line.split(',')
            image_name = tokens[0].split('/')[-1]
            # date = tokens[1]
            # lat = tokens[2]
            # lon = tokens[3]
            # zoom = tokens[4]
            # user = tokens[5]
            # title = tokens[6]
            # description = tokens[7]

            path_map[image_name] = tokens

    return path_map

last_time = time.time()


for month in months:
    f = h5py.File(join(output_dir, month + '.h5'), 'w')

    sub_image_names = os.listdir(join(image_dir, month, month))

    path_map = load_path_map(join(paths_dir, month))
    valid_im_names = [x for x in sub_image_names if x in path_map]

    image_names = [path_map[x][0].encode("ascii", "ignore") for x in valid_im_names]
    dates = [path_map[x][1].encode("ascii", "ignore") for x in valid_im_names]
    lats = [path_map[x][2].encode("ascii", "ignore") for x in valid_im_names]
    lons = [path_map[x][3].encode("ascii", "ignore") for x in valid_im_names]
    zooms = [path_map[x][4].encode("ascii", "ignore") for x in valid_im_names]
    users = [path_map[x][5].encode("ascii", "ignore") for x in valid_im_names]
    titles = [path_map[x][6].encode("ascii", "ignore") for x in valid_im_names]
    descriptions = [path_map[x][7].encode("ascii", "ignore") for x in valid_im_names]

    numpy_bottleneck = np.load(join(bottleneck_dir, month + '.npy'))
    bottleneck = f.create_dataset('bottlenecks', data=numpy_bottleneck)


    image_name = f.create_dataset('image_names', data=image_names)
    date = f.create_dataset('dates', data=dates)
    lat = f.create_dataset('lats', data=dates)
    lon = f.create_dataset('lons', data=dates)
    zoom = f.create_dataset('zooms', data=zooms)
    user = f.create_dataset('users', data=users)
    titles = f.create_dataset('titles', data=titles)
    descriptions = f.create_dataset('descriptions', data=descriptions)

    f.close()

    this_time = time.time()
    print('Finished {} in {} seconds'.format(month, time.time() - last_time))
    last_time = this_time












