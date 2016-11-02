import sys
from subprocess import call
from os.path import join
from glob import glob


image_dir = '/tigress/dchouren/thesis/images/'

def download_file(filename):
    print('Starting {}'.format(filename))
    with open(filename, 'r') as inf:
        for line in inf:
            url = line.split(',')[0]
            image_name = url.split('/')[-1]

            return_code = call(['wget', '-t', '3', '-T', '5', '--quiet', url, '-O', join(image_dir, image_name)])
            if return_code == 0:
                print('Success: {}'.format(url))
            else:
                print('Error: {}'.format(url))

    print('Done {}'.format(filename))


def download_dir(dirname):
    filenames = glob(join(dirname, '*'))
    for filename in filenames:
        download_file(filename)


if __name__ == '__main__':
    dirname = sys.argv[1]

    download_dir(dirname)
