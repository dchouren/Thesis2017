#!/usr/bin/env bash

set -e

if [ $# -ne 3 ]; then
    echo $0: usage: classify_photos.sh [GEO_ID] [BATCH_SIZE] [MODEL]
    echo $1: example: classify_photos.sh 187147 100 5 gold_model
    echo $2: geo_id - which geo you want to classify restaurant photos for
    echo $3: batch_size - how many photos to download and process at once
    echo $4: model - name of the model in trained_models that you want to use
    exit 1
fi

pushd () {
    command pushd "$@" > /dev/null
}

BASE_DIR=~/vision

GEO_ID=$1
HDF5_DIR=${BASE_DIR}/h5/${GEO_ID}
OUTPUT_DIR=${BASE_DIR}/output/${GEO_ID}

if [ ! -d "$HDF5_DIR" ]; then
    mkdir $HDF5_DIR
fi
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir $OUTPUT_DIR
fi
if [ ! -d "$BASE_DIR/images" ]; then
    mkdir $BASE_DIR/images
fi

pushd ${BASE_DIR}


FPATHS=resources/${GEO_ID}/fpaths_lua
PATHS=resources/${GEO_ID}/paths

# get image paths
#hive -hiveconf GEOID=${GEO_ID} -e 'select m.path from t_media m join t_media_locations l on m.id=l.mediaid where l.locationid in (select p.locationid from t_locationpaths p join t_location t on p.locationid=t.id where p.parentid=${hiveconf:GEOID} and t.placetypeid=10022)' > ${PATHS}


BATCH_SIZE=$2
RATE_LIMIT=$3
MODEL=$4

NUM_PATHS=$(cat ${PATHS} | wc -l)
echo ${NUM_PATHS}
COUNTER=${BATCH_SIZE}
LOOP_COUNT=0
echo ${HDF5_FILE}
while [ $COUNTER -lt $[${NUM_PATHS}+${COUNTER}] ]; do
    echo ${COUNTER}
    rm -f images/* 2> /dev/null
    rm -f ${HDF5_FILE} 2> /dev/null

    pushd ${BASE_DIR}
    echo "Downloading images"
    echo ""
    head -n ${COUNTER} ${PATHS} | tail -n $BATCH_SIZE | python3 src/old_format_image_paths.py ${FPATHS}
    cat ${FPATHS} | cut -d ',' -f 1 | parallel --gnu "wget -P images2/ {} || echo 'ok'"

    # convert images to same size
    SIZE=100
    pushd ${BASE_DIR}/images2
    echo "Resizing images to ${SIZE}x${SIZE}"
    echo ""
    for f in *.jpg*; do
      convert -resize ${SIZE}x${SIZE}! ./"$f" ./"$f"
    done


    # convert images to hdf5
    echo "Converting images to HDF5"
    echo ""
    pushd ${BASE_DIR}

    HDF5_FILE=${HDF5_DIR}/hdf5
    python3 src/convert_to_hdf5.py ${HDF5_FILE} ${FPATHS} images2


    # run model
    echo "Predicting"
    echo ""
    pushd model
    OUTPUT_FILE=${OUTPUT_DIR}/${LOOP_COUNT}
    th predict.lua -m ../trained_models/${MODEL} -f ${HDF5_FILE} -s ${OUTPUT_FILE}
    echo "Outputted to ${OUTPUT_FILE}"

#    for FILE in ${HDF5_DIR}/*; do
#        OUTPUT_FILE=${OUTPUT_DIR}/$(basename ${FILE} .${FILE##*.})
#        th predict.lua -m ../trained_models/${MODEL} -f ${FILE} -s ${OUTPUT_FILE}
#        echo "Outputted to ${OUTPUT_FILE}"
#    done

    COUNTER=$[${COUNTER}+${BATCH_SIZE}]
    LOOP_COUNT=$[${LOOP_COUNT}+1]
#    sleep $RATE_LIMIT
done



echo "### DONE ###"