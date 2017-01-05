import glob
import ipdb
import numpy as np
import pickle
from keras.preprocessing import image
from os.path import join
from vgg19 import VGG19

from src._keras.models.imagenet_utils import preprocess_input

# input_dir = sys.argv[1]

MENU = 922


image_files = glob.glob(join(input_dir, '*.jpg*'))
loaded_images = [image.load_img(image_file, target_size=(224,224)) for image_file in image_files]
image_arrays = [image.img_to_array(img) for img in loaded_images]
image_arrays = [np.expand_dims(x, axis=0) for x in image_arrays]
image_arrays = [preprocess_input(x) for x in image_arrays]
X = np.vstack(image_arrays)

with open('imagearrays.pickle', 'wb') as outf:
    pickle.dump(X, outf)


# model = ResNet50(weights='imagenet')
model = VGG19(weights='imagenet')

from keras.optimizers import SGD
model.compile(loss='sparse_categorical_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

# ipdb.set_trace()
y = np.asarray([MENU]*X.shape[0])


model.fit(X,y, nb_epoch=5, batch_size=32)
with open('newmodel.pickle', 'wb') as outf:
    pickle.dump(model, outf)

ipdb.set_trace()


