'''
Convert images in a given fpath file to a numpy array. Programmatically splits npy arrays and saves to a base directory.

usage: python images_to_numpy.py [image_dir] [output_base] [fpath_file]
'''

import sys

from os.path import join
import pickle
import glob

import numpy as np

from vision_utils import load_and_preprocess_image



def get_fpath_filename(filepath):
    return filepath.split('photo-s/')[-1].replace('{','').replace('}','').replace('/','-')

image_dir = sys.argv[1]
output_base = sys.argv[2]
fpath_file = sys.argv[3]

x_size, y_size = 224, 224

num_images = 0
count = 0
preprocessed_images = []


dataset = image_dir.split('/')[-1]
images = glob.glob(join(image_dir, '*'))

with open(fpath_file, 'r') as inf:
    for line in inf:
        tokens = line.strip().split(',')
        url = tokens[0].replace('{','').replace('}','')
        mediaid = tokens[1]
        locationid = tokens[2]
for image_path in images:

    im_path = join(image_dir, image_path)

    x = load_and_preprocess_image(im_path, x_size, y_size, True)
    if x is False:
        print('Failed on {}'.format(im_path))
        continue

    num_images += 1
    if num_images % 1000 == 0:
        print(num_images)

    preprocessed_images.append(x)

    batch_size = 50000
    if num_images % batch_size == 0:
        np_images = np.asarray(preprocessed_images)

        output_file = join(output_base, str(count) + '.npy')
        with open(output_file, 'wb') as outf:
            np.save(outf, preprocessed_images)
        print('Wrote {} images to {}'.format(batch_size, output_file))

        count += 1
        preprocessed_images = []


np_images = np.asarray(preprocessed_images)

output_file = join(output_base, str(count) + '.npy')
with open(output_file, 'wb') as outf:
    np.save(outf, preprocessed_images)
print('Wrote {} images to {}'.format(batch_size, output_file))


