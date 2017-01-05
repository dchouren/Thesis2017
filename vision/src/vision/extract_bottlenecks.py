'''
Run images through a topless base model and save features for future predicting/training. You need to put the image directory you want to save in a wrapper directory. Eg, if you want to save san_francisco images, you need to hide your images in $IMAGES/testsets/wrapper/san_francisco, where san_francisco is the only directory in wrapper/.

usage: python extract_bottlenecks.py im_dir output model_name
example: python src/vision/extract_bottlenecks.py $IMAGES/testsets/wrapper $RESOURCES/bottlenecks/testsets/san_francisco resnet50
'''

import sys
import os
import glob

from bottleneck_train import save_bottleneck_features
from models.utils import _load_model


if len(sys.argv) != 4:
    print (__doc__)
    sys.exit(0)

im_dir = sys.argv[1]
output = sys.argv[2]
model_name = sys.argv[3]

model = _load_model(model_name, include_top=False)

labels = sorted(os.listdir(im_dir))
class_sizes = [len(os.listdir(os.path.join(im_dir, label))) for label in labels]

nb_samples = sum(class_sizes)

save_bottleneck_features(model, im_dir, (224, 224), 256, nb_samples, output)
print('Saved {} images to {}'.format(nb_samples, output))
