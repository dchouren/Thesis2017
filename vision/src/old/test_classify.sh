#!/usr/bin/env bash

set -e

if [ $# -ne 4 ]; then
    echo $0: usage: test_classify.sh [MODEL] [SECOND_MODEL] [GEO_ID] [BATCH_SIZE]
    echo $1: example: ./src/test_classify.sh vgg16 ./trained_models/vgg16/finetuned_50_3.h5 187147 10000
    echo $2: example: ./src/test_classify.sh inception ./trained_models/inception/finetuned_50_3.h5 187147 10
    echo $3: model - name of the model in trained_models that you want to use [resnet50, vgg16, vgg19]
    echo $4: geo_id - which geo you want to classify restaurant photos for
    echo $5: batch_size - how many photos to download and process at once
    exit 1
fi

pushd () {
    command pushd "$@" > /dev/null
}

BASE_DIR=$(pwd)
SCRATCH_DIR=/media/storage/scratch

MODEL=$1
SECOND_MODEL=$2
GEO_ID=$3
BATCH_SIZE=$4

HDF5_DIR=${SCRATCH_DIR}/h5/${GEO_ID}
OUTPUT_DIR=${BASE_DIR}/output/${MODEL}/${GEO_ID}
IMAGES_DIR=${BASE_DIR}/images

#if [ ! -d "$HDF5_DIR" ]; then
#    sudo mkdir $HDF5_DIR
#fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi
if [ ! -d "$IMAGES_DIR" ]; then
    sudo mkdir IMAGES_DIR
fi


echo "Output dir is ${OUTPUT_DIR}"


pushd ${BASE_DIR}

# download images
#head -n 100 ${BASE_DIR}/resources/paths | python3 ${BASE_DIR}/src/format_image_paths.py | parallel --gnu "wget -P ${BASE_DIR}/images/ {}"

FPATHS_DIR=resources/${GEO_ID}/fpaths
PATHS=resources/${GEO_ID}/paths


#hive -hiveconf GEOID=${GEO_ID} -e 'select m.path, m.id, l.locationid from t_media m join t_media_locations l on m.id=l.mediaid where l.locationid in (select p.locationid from t_locationpaths p join t_location t on p.locationid=t.id where p.parentid=${hiveconf:GEOID} and t.placetypeid=10022) order by l.locationid' > ${PATHS}

echo "Formatting paths"
rm ${FPATHS_DIR}/* || true
cat ${PATHS} | python src/format_image_paths.py ${FPATHS_DIR} ${BATCH_SIZE}

NUM_PATHS=$(cat ${PATHS} | wc -l)
echo ${NUM_PATHS}
COUNTER=${BATCH_SIZE}
#COUNTER=410000
LOOP_COUNT=$[ ((${COUNTER} / ${BATCH_SIZE}))  -1 ]
#echo ${HDF5_FILE}
while [ $COUNTER -lt $[${NUM_PATHS}+${BATCH_SIZE}] ]; do
    echo ${COUNTER}

    find ${IMAGES_DIR} -name '*.jpg' -print0 | xargs -0 rm || true
    FPATHS_FILE=${FPATHS_DIR}/${LOOP_COUNT}

    echo "Downloading images"
    echo ""
    echo ${FPATHS_FILE}
    cat ${FPATHS_FILE} | cut -d ',' -f 1 | parallel curl -s -o ${IMAGES_DIR}/#1-#2-#3-#4-#5 || echo "Failed"

    # run model
    echo "Predicting"
    echo ${COUNTER}
    echo ""
#    pushd model
    OUTPUT_FILE=${OUTPUT_DIR}/$(printf "%05d" ${LOOP_COUNT})
#    th predict.lua -m ../trained_models/${MODEL} -f ${HDF5_FILE} -s ${OUTPUT_FILE} -c ${CUDA}
    python3 src/test.py ${MODEL} ${SECOND_MODEL} ${FPATHS_FILE} ${IMAGES_DIR} ${OUTPUT_FILE}
    echo "Outputted to ${OUTPUT_FILE}"

    COUNTER=$[${COUNTER}+${BATCH_SIZE}]
    LOOP_COUNT=$[((${COUNTER} / ${BATCH_SIZE}))-1]
done

popd



echo "### DONE ###"
