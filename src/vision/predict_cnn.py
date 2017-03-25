import sys
from os.path import join

from models.utils import _load_model
from keras.models import load_model

import ipdb


saved_weights = sys.argv[1]

model_dir = '/tigress/dchouren/thesis/trained_models'

ipdb.set_trace()

# nadam = load_model(join(model_dir, 'resnet50_10_5_nadam'))
resnet50 = _load_model('resnet50', include_top=True, weights=None)
resnet50.load_weights(join(model_dir, saved_weights), by_name=True)

resnet50.save(join(model_dir, 'test.model'))