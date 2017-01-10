'''
Usage: python predict_thumbnail.py [base_model] [top_model] [im_dir] [formatted_paths_file] [output_file] [mapping]
Production: python src/vision/predict_thumbnail.py vgg16 /scratch/resources/trained_models/thumbnail/top_vgg16_100_55707_dslr.h5 /scratch/dchouren/images/testsets/10022/paris resources/fpaths/geos_10022/paris output/thumbnail/dslr/paris dslr
'''

import cv2
import sys
import time
from os.path import join

import quality_utils
from _utils import get_fpath_filename
from vision_utils import load_and_preprocess_image, preprocess_input, _decode_predictions

from keras.models import load_model
from models.utils import _load_model

import ipdb

if len(sys.argv) < 6:
    print (__doc__)
    sys.exit(0)

base_model_path = sys.argv[1]
top_model_path = sys.argv[2]
im_dir = sys.argv[3]
formatted_paths_file = sys.argv[4]
output_file = sys.argv[5]
mapping = sys.argv[6]


start_time = time.time()

base_model = _load_model(base_model_path, include_top=False)
menu_model = load_model('/scratch/resources/trained_models/menu/menu_receipt_classifier.h5')
top_model = load_model(top_model_path)
face_cascade = cv2.CascadeClassifier('/scratch/resources/xml/haarcascade_frontalface_default.xml')

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


entropy_threshold = 4.5
digital_count = 0
face_count = 0
menu_count = 0

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


        # Load using CV2 to do some face detection
        # Can also use quality_utils to get some blurriness metrics
        cv_im = cv2.imread(im_path, 0)
        if cv_im is None:
            continue

        has_face = False
        is_digital = False
        is_textlike = False

        if quality_utils.has_face(cv_im, face_cascade):
            # print('has face: {}'.format(url))
            face_count += 1
            prediction = 'has_face'
            probability = 1.0
            has_face = True
        if quality_utils.get_entropy(cv_im) <= entropy_threshold:
            # print('digital: {}'.format(url))
            digital_count += 1
            prediction = 'digital'
            probability = 1.0
            is_digital = True

        x = load_and_preprocess_image(im_path, x_size, y_size, True)
        if x is False:
            continue

        base_features = base_model.predict(x)
        menu_preds = menu_model.predict(base_features)
        menu_pred, menu_prob = _decode_predictions(menu_preds, 'menu')[0]

        if menu_pred != 'other':
            prediction, probability = menu_pred, menu_prob
            menu_count += 1
            is_textlike = True

        preds = top_model.predict(base_features)
        prediction, probability = _decode_predictions(preds, mapping)[0]
        probability = float(probability)


        tokens.extend([str(has_face), str(is_digital), str(is_textlike), prediction, str(probability)])

        output = ','.join([str(x) for x in tokens]) + '\n'
        outf.write(output)

        num_images += 1


print()
print('{} images with faces'.format(face_count))
print('{} digital images'.format(digital_count))
print('{} menu images'.format(menu_count))
print('Took {} seconds'.format(time.time() - start_time))
print('Output is {}'.format(output_file))
