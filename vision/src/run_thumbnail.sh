#!/usr/bin/env bash

CONDA_ENV=daneel
source activate ${CONDA_ENV}

TASK=thumbnail
DATASET=dslr

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

echo "Pull photo information"
ORIG_PHOTO_FILE=resources/paths/thumbnail/dslr/camera

psql -h tripmonster.tripadvisor.com -U tripmonster -A -F , -X -t -f resources/queries/camera.psql -o ${ORIG_PHOTO_FILE}

rm ${PATHS_DIR}/dslr || true
rm ${PATHS_DIR}/nondslr || true
grep "NIKON D.*\|Canon EOS.*" $ORIG_PHOTO_FILE >> ${PATHS_DIR}/dslr
grep -v "NIKON D.*\|Canon EOS.*"  $ORIG_PHOTO_FILE >> ${PATHS_DIR}/nondslr

#rm $ORIG_PHOTO_FILE || true

./pull_train_val.sh ${TASK} ${DATASET} dslr nondslr
./pull_testsets.sh 10022

# train top model from bottleneck info
echo "Training top model on bottleneck features"
MODEL=vgg16
NUM_EPOCHS=100
BATCH_SIZE=256
MAPPING=dslr
python src/vision/bottleneck_train.py ${MODEL} ${TASK} ${DATASET} ${NUM_EPOCHS} ${BATCH_SIZE} ${MAPPING}

MODEL_DIR=/spindle/${USER}/resources/${TASK}/${DATASET}/trained_models/
TOP_MODEL=${MODEL_DIR}/top_${MODEL}_${TASK}_${DATASET}_${NUM_EPOCHS}_${MAPPING}.5

# finetune last block
echo "Finetuning"
NUM_FINETUNE_EPOCHS=200
SAMPLES_PER_EPOCH_FACTOR=2  # factor to downsample by
python src/vision/finetune.py ${MODEL} ${TASK} ${DATASET} ${TOP_MODEL} ${NUM_FINETUNE_EPOCHS} ${SAMPLES_PER_EPOCH_FACTOR} ${BATCH_SIZE} dslr

# the finetuned model is named programmatically
FINETUNED_MODEL=${MODEL_DIR}/finetuned_${MODEL}_${TASK}_${DATASET}_${NUM_EPOCHS}_${MAPPING}.h5

FINAL_RESULTS_FILE=thumbnail_results.csv
mv ${FINAL_RESULTS_FILE} old_${FINAL_RESULTS_FILE} || true
echo "Final results will be written to ${FINAL_RESULTS_FILE}"


if [ ! -e thumbnail_results ]
then
    mkdir -p thumbnail_results
fi

# predict for testsets
GEOS=( "milan" "new_york" "paris" )
GEOIDS=( 187849 60763 187147 )
for ((i=0;i<${#GEOS[@]};++i)); do
    GEO=${GEOS[i]}
    IMAGE_DIR=/spindle/${USER}/images/testsets/${GEO}
    echo "Predicting ${GEO}"

    MODEL_OUTPUT=output/${TASK}/${DATASET}/finetuned_menu_face_dslr_${GEO}
    python src/vision/predict_thumbnail.py ${FINETUNED_MODEL} resources/fpaths/${TASK}/${GEO} ${IMAGE_DIR} ${MODEL_OUTPUT} dslr

    # organize_ditto_results formats two internal model results with ditto results. Here I'm just showing one run of predict_thumbnail.py and using those results twice for organize_ditto_results. For our live site test, I had played with some of the parameters predict_thumbnail to generate a different set of internal results.
    python src/ditto/organize_ditto_results.py resources/fpaths/geos_10022/${GEO} thumbnail_results/${GEO}_results.csv ditto_results/good_ditto_${GEO}.json ${MODEL_OUTPUT} ${MODEL_OUTPUT}   # replace the duplicated MODEL_OUTPUT with results from another test, or modify organize_ditto_results to take however many result files as you want

    cat ${GEO}_results.csv >> thumbnail_results/${FINAL_RESULTS_FILE}

    # examine results for a geo
    python src/html/compare_results.py ${GEO}_results.csv html/${TASK}/${DATASET}/${GEO}_compare.html ${GEO}
done

echo "Final results are in thumbnail_results/${FINAL_RESULTS_FILE}"
echo "Done"