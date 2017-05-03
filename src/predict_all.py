import os
from os.path import join
from glob import glob



model_dir = '/tigress/dchouren/thesis/trained_models/'



model_files = glob(join(model_dir, '*weights*.h5'))


checkpointed_model = {}
for model_file in model_files:
    base_model_name = model_file.split('_weights')[0]
    if base_model_name in checkpointed_model:
        checkpointed_model[base_model_name] = checkpointed_model
    else:
        checkpointed_model[base_model_name] = [checkpointed_model]
    