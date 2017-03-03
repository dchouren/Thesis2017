import sys
import glob
import h5py
import os

import numpy as np

from os.path import join

import ipdb

bottleneck_dir = sys.argv[1]
output_dir = sys.argv[2]
image_dir = sys.argv[3]
paths_dir = sys.argv[4]

bottlenecks = glob.glob(join(bottleneck_dir, '*.npy'))

months = ['01','02','03','04','05','06','07','08','09','10','11','12']

def load_path_map(path_file):
    path_map = {}
    with open(join(path_file), 'r') as inf:
        for line in inf:
            tokens = line.split()
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


for month in months:
    f = h5py.File(join(output_dir, month + '.h5'), 'w')

    sub_image_names = os.listdir(join(image_dir, month, month))

    path_map = load_path_map(join(paths_dir, month))
    image_names = [path_map[x][0] for x in sub_image_names if x in path_map]
    dates = [path_map[x][1] for x in sub_image_names if x in path_map]
    lats = [path_map[x][2] for x in sub_image_names if x in path_map]
    lons = [path_map[x][3] for x in sub_image_names if x in path_map]
    zooms = [path_map[x][4] for x in sub_image_names if x in path_map]
    users = [path_map[x][5] for x in sub_image_names if x in path_map]
    titles = [path_map[x][6] for x in sub_image_names if x in path_map]
    descriptions = [path_map[x][7] for x in sub_image_names if x in path_map]

    numpy_bottleneck = np.load(join(bottleneck_dir, month + '.npy'))
    bottleneck = f.create_dataset('bottlenecks', data=numpy_bottleneck)

    ipdb.set_trace()

    image_name = f.create_dataset('image_names', data=image_names)
    date = f.create_dataset('dates', data=dates)
    lat = f.create_dataset('lats', data=dates)
    lon = f.create_dataset('lons', data=dates)
    zoom = f.create_dataset('zooms', data=zooms)
    user = f.create_dataset('users', data=users)
    titles = f.create_dataset('titles', data=titles)
    descriptions = f.create_dataset('descriptions', data=descriptions)

    f.close()












