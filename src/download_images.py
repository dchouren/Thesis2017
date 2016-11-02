import sys
import os
from subprocess import call
from os.path import join
from glob import glob


image_dir = '/tigress/dchouren/thesis/images/'

def get_leaf_dir(dirname):
    return dirname.split('/')[-1]


def download_file(label, filename):
    print('Starting {}'.format(filename))

    with open(filename, 'r') as inf:
        for line in inf:
            url = line.split(',')[0]
            image_name = url.split('/')[-1]

            return_code = call(['wget', '-t', '3', '-T', '5', '--quiet', url, '-O', join(image_dir, label, image_name)])
            if return_code == 0:
                print('Success: {}'.format(url))
            else:
                print('Error: {}'.format(url))

    print('Done {}'.format(filename))


def download_dir(dirname):
    label = get_leaf_dir(dirname)
    if not os.path.isdir(join(image_dir, label)):
        os.mkdir(join(image_dir, label))

    filenames = sorted(glob(join(dirname, '*')))
    for filename in filenames:
        download_file(label, filename)


if __name__ == '__main__':
    dirname = sys.argv[1]

    download_dir(dirname)
