from src.models.utils import _load_model
from keras.preprocessing import image
from keras.models import load_model
from src.models.imagenet_utils import preprocess_input, decode_predictions, _decode_predictions, load_and_preprocess_image

import numpy as np
import glob
import sys
import time
from os.path import join

# from resnet import read_img_batch_tuples, predict_batch

import ipdb

def get_filename(filepath):
    return filepath.split('photo-s/')[-1].replace('/','-')


model_name = sys.argv[1]
second_model = sys.argv[2]
formatted_paths_file = sys.argv[3]
im_dir = sys.argv[4]
output_file = sys.argv[5]

start_time = time.time()

model = _load_model(model_name, weights='imagenet')
base_model = _load_model(model_name, include_top=False)
second_model = load_model(second_model)
# model.load_weights('./resources/weights/vgg16/fine-tuned-weights.h5', by_name=True)

# img_paths = glob.glob('../resources/f5_images/*.jpg*')

x_size, y_size = 224, 224
if model_name == 'inception':
    x_size, y_size = 299, 299

num_menu = 0
num_images = 0
num_filtered = 0


outf = open(output_file, 'w')
debug_outf = open('debug/' + output_file.split('/')[-1], 'w')

print(formatted_paths_file)

with open(formatted_paths_file, 'r') as inf:
    # fpaths = inf.readlines()

    # img_batch, meta_batch = read_img_batch_tuples(fpaths)
    # ipdb.set_trace()
    # predictions = predict_batch(model, img_batch)

    seen_paths = {}


    for line in inf:
        num_images += 1

        url, mediaid, locationid, memberid, date, ownerid, isowner, points = line.strip().split(',')
        url = url.replace('{','').replace('}','')
        date = date.split(' ')[0] # drop the H:M:S part

        im_name = get_filename(url)
        # num_seen = 0
        # try:
        #     num_seen = seen_paths[im_name]
        #     seen_paths[im_name] += 1
        #     im_name += '.{}'.format(num_seen)
        #     try:
        #         seen_paths[im_name] += 1
        #     except:
        #         seen_paths[im_name] = 1
        # except:
        #     seen_paths[im_name] = 1


        im_path = join(im_dir, im_name)
        x = load_and_preprocess_image(im_path, x_size, y_size, False)
        if x is False:
            continue

        # ipdb.set_trace()

        preds = model.predict(x)

    # for preds, meta in zip(predictions, meta_batch):
        prediction = decode_predictions(preds)[0][1]
        probability = max(preds[0])

        if prediction == 'menu':
            x *= 1./255   # second model is trained on rescaled data

            base_preds = base_model.predict(x)
            new_preds = second_model.predict(base_preds)[0]
            prediction, probability = _decode_predictions(new_preds, 'three_class')

            if prediction == 'receipt' or prediction == 'menu':
                num_filtered += 1

            debug_outf.write('{},{},{},{},{},{},{},{},{},{}\n'.format(im_path, mediaid, locationid, memberid, date, ownerid, isowner, points, prediction, probability))

        # prob = model.predict_proba(x)

        output = '{},{},{},{},{},{},{},{},{},{}\n'.format(url, mediaid, locationid, memberid, date, ownerid, isowner, points, prediction, probability)

        outf.write(output)
        # print()
        # print(im_file)
        # print('Predicted: ', prediction)

        if prediction == 'menu':
            num_menu += 1

print()
print('Predicted {} out of {} as menu. Filtered {} initial menu predictions'.format(num_menu, num_images, num_filtered))
print('Took {} seconds'.format(time.time() - start_time))
