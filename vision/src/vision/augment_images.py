'''
Augment images with shears, etc

Usage: python augment_images.py [input_dir] [output_dir] [save_prefix] [num_augments]

I never got num_augments to work quite right
'''

import sys
import glob
from os.path import join

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import ipdb

input_dir = sys.argv[1]
output_dir = sys.argv[2]
save_prefix = sys.argv[3]
num_augments = int(sys.argv[4])

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


images = glob.glob(join(input_dir, '*'))

processed_images = []

for im in images:

    img = load_img(im)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    processed_images.append(x)

print(len(processed_images))
for x in processed_images:
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=output_dir, save_prefix=save_prefix, save_format='jpg'):
        i += 1
        if i > num_augments:
            break  # otherwise the generator would loop indefinitely