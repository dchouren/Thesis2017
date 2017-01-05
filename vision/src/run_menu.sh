#!/usr/bin/env bash

CONDA_ENV=daneel
source activate ${CONDA_ENV}

TASK=menu
DATASET=menu_receipt

echo "Task is ${TASK}"
echo "Dataset is ${DATASET}"

ROOT_DIR=~/vision


PATHS_DIR=${ROOT_DIR}/resources/paths/${TASK}/${DATASET}
FPATHS_DIR=${ROOT_DIR}/resources/fpaths/${TASK}/${DATASET}

if [ ! -e ${PATHS_DIR} ]
then
    sudo mkdir ${PATHS_DIR}
fi
if [ ! -e ${FPATHS_DIR} ]
then
    sudo mkdir ${FPATHS_DIR}
fi

echo "Train and val sets were manually created using label_data.py"

./pull_testsets.sh 10022

# train top model from bottleneck info
echo "Training top model on bottleneck features"
MODEL=vgg16
NUM_EPOCHS=100
BATCH_SIZE=256
MAPPING=menu
#python src/vision/bottleneck_train.py ${MODEL} ${TASK} ${DATASET} ${NUM_EPOCHS} ${BATCH_SIZE} ${MAPPING}

MODEL_DIR=/spindle/${USER}/resources/${TASK}/${DATASET}/trained_models/
TOP_MODEL=${MODEL_DIR}/top_${MODEL}_${TASK}_${DATASET}_${NUM_EPOCHS}_${MAPPING}.h5

# finetune last block
echo "Finetuning"
NUM_FINETUNE_EPOCHS=2
SAMPLES_PER_EPOCH_FACTOR=2  # factor to downsample by
python src/vision/finetune.py ${MODEL} ${TASK} ${DATASET} ${TOP_MODEL} ${NUM_FINETUNE_EPOCHS} ${SAMPLES_PER_EPOCH_FACTOR} ${BATCH_SIZE} ${MAPPING}

# the finetuned model is named programmatically
FINETUNED_MODEL=${MODEL_DIR}/finetuned_${MODEL}_${TASK}_${DATASET}_${NUM_EPOCHS}_${MAPPING}.h5



# predict for testsets
GEOS=( "milan" "new_york" "paris" )
GEOIDS=( 187849 60763 187147 )
for ((i=0;i<${#GEOS[@]};++i)); do
    GEO=${GEOS[i]}
    IMAGE_DIR=/spindle/${USER}/images/testsets/${GEO}
    echo "${GEO}"

    MODEL_OUTPUT=output/${TASK}/${DATASET}/finetuned_menu_${GEO}
    python src/vision/predict_menu.py ${FINETUNED_MODEL} resources/fpaths/${TASK}/${GEO} ${IMAGE_DIR} ${MODEL_OUTPUT} ${MAPPING}
done

echo "Done"