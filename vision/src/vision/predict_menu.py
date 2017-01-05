'''
Usage: python test.py [base_model] [menu_model] [im_dir] [formatted_paths_file] [output_file] [mapping] [label]
Production: python src/vision/predict_menu.py vgg16 /spindle/resources/trained_models/menu/menu_receipt_classifier.h5 /spindle/dchouren/images/testsets/10022/paris resources/fpaths/geos_10022/paris output/menu/paris menu
'''

import cv2
import sys
import time
from keras.models import load_model
from models.utils import _load_model

from os.path import join

from _utils import get_fpath_filename
from quality_utils import *
from vision_utils import load_and_preprocess_image, preprocess_input, _decode_predictions

import ipdb


if len(sys.argv) < 6:
    print (__doc__)
    sys.exit(0)

base_model_path = sys.argv[1]
menu_model_path = sys.argv[2]
im_dir = sys.argv[3]
formatted_paths_file = sys.argv[4]
output_file = sys.argv[5]
mapping = sys.argv[6]

start_time = time.time()

base_model = _load_model(base_model_path, include_top=False)
menu_model = load_model(menu_model_path)

x_size, y_size = 224, 224
if 'inception' in base_model_path:
    print('Detected inception model. Changing default image size from (224, 224) to (299, 299)')
    x_size, y_size = 299, 299

num_label = 0
num_images = 0

outf = open(output_file, 'w')
print('Output is {}'.format(output_file))
debug_outf = open('debug/' + output_file.split('/')[-1], 'w')

print('Reading from {}'.format(formatted_paths_file))
print('Image dir: {}'.format(im_dir))


last_time = time.time()

with open(formatted_paths_file, 'r') as inf:

    for line in inf:
        if num_images % 1000 == 0:
            print(num_images, time.time() - last_time)
            last_time = time.time()

        tokens = line.strip().split(',')
        url = tokens[0]
        im_name = get_fpath_filename(url)

        im_path = join(im_dir, im_name)
        url = url.replace('{','').replace('}','')
        tokens[0] = url

        # Can also use quality_utils to get some blurriness metrics
        cv_im = cv2.imread(im_path, 0)
        if cv_im is None:
            continue
        blurriness = get_blurriness(get_laplacian(cv_im))


        x = load_and_preprocess_image(im_path, x_size, y_size, True)
        if x is False:
            continue

        base_features = base_model.predict(x)
        menu_preds = menu_model.predict(base_features)
        menu_pred, menu_prob = _decode_predictions(menu_preds, mapping)[0]

        tokens.extend([blurriness, menu_pred, str(menu_prob)])

        output = ','.join([str(x) for x in tokens]) + '\n'
        outf.write(output)

        num_images += 1



print()
print('Took {} seconds'.format(time.time() - start_time))
print('Output is {}'.format(output_file))
