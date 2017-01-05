import sys
import os
from subprocess import call
import subprocess
from os.path import join
from glob import glob

import ipdb

# python3 src/im_download.py ~/vision/resources/fpaths/thumbnail/flickr /media/storage/scratch/dchouren_images/thumbnail/flickr/train/34542434@N00/

def get_leaf_dir(dirname):
    return dirname.split('/')[-1]


def download_file(label, filename):
    print('Starting {}'.format(filename))

    with open(filename, 'r') as inf:
        for line in inf:
            url = line.split(',')[0].strip()
            image_name = url.split('/')[-1]

            proc = subprocess.Popen(['wget', '-t', '3', '-T', '5', '--wait=0.0', '--quiet', url, '-O', join(image_dir,
                                                                                                 image_name)],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            output, err = proc.communicate()
            print(output)
            print(err)
            if proc == 0:
                print('Success: {}'.format(url))
            else:
                with open('errors', 'a') as errorf:
                    errorf.write(line)
                print('Error: {}'.format(url))

    print('Done {}'.format(filename))


def download_dir(dirname):
    label = get_leaf_dir(dirname)
    # if not os.path.isdir(join(image_dir, label)):
    #     os.mkdir(join(image_dir, label))

    filenames = sorted(glob(join(dirname, '*')))
    for filename in filenames:
        download_file(label, filename)


if __name__ == '__main__':
    fpath = sys.argv[1]
    image_dir = sys.argv[2]

    download_file(fpath)