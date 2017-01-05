import datetime
import numpy as np
from PIL import Image
from fn.func import F
from fn.uniform import *

from src.keras_utils import timeit

MODEL_WEIGHTS = 'imagenet'
BATCH_SIZE = 512



def load_img(path, target_size=(224, 224)):
    return Image.open(path).resize((target_size))


def img_to_array(img, tp=(2, 0, 1)):
    return np.asarray(img, dtype="float32").transpose(tp)


def preprocess_image(img):
    img[0, :, :] -= 103.939
    img[1, :, :] -= 116.779
    img[2, :, :] -= 123.68
    # 'RGB'->'BGR'
    return img[::-1, :, :]


### This is a composition of several functions, equivalent to but with nice
### sytax.
### def load_and_process_image(filename):
### return preprocess_image(img_to_array(load_img(filename)))

load_and_process_image = F(preprocess_image) << F(img_to_array) << F(load_img)



def problem_img_log(msg, fn="./problem.log"):
    """logs problems to a file."""
    with open(fn, 'a') as f:
        msg = "{} : {}".format(datetime.datetime.now(), msg)
        f.writelines(msg + "\n")


@timeit
def read_img_batch(file_paths):
    return np.array([load_and_process_image(img) for img in file_paths])

def get_filename(filepath):
    return filepath.split('/')[-1]


@timeit
def read_img_batch_tuples(fpaths):
    img_batch = []
    meta_batch = []
    for fpath in fpaths:
        tokens = fpath.split(',')
        try:
            im_path = 'images/' + get_filename(tokens[0])
            img_batch.append(load_and_process_image(im_path))
            meta_batch.append(tokens)
        except:
            print("failed to read or load {}".format(tokens[0]))
            problem_img_log(msg="file {} failed to read".format(tokens[0]))
            pass

    return img_batch, meta_batch


@timeit
def predict_batch(model, data):
    return model.predict(data, batch_size=BATCH_SIZE)




