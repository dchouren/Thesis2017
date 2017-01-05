#!/usr/bin/env bash
if [ "$#" -lt 3 ]; then
    echo "Need at least three parameters: task, dataset, and labels"
fi

TASK=$1
DATASET=$2
LABELS="${@:3}"

ROOT_DIR=~/vision

PATHS_DIR=${ROOT_DIR}/resources/paths/${TASK}/${DATASET}
FPATHS_DIR=${ROOT_DIR}/resources/fpaths/${TASK}/${DATASET}


IMAGE_DIR=/spindle/dchouren/images/${TASK}/${DATASET}
TRAIN=${IMAGE_DIR}/train
VAL=${IMAGE_DIR}/val


for label in "${LABELS[@]}"; do
    echo $label
    python src/tools/format_image_paths.py ${PATHS_DIR} ${FPATHS_DIR} ${label} -1

    if [ ! -e ${IMAGE_DIR}/train/${label} ]
    then
        mkdir -p ${IMAGE_DIR}/train/${label}
    fi
    if [ ! -e ${IMAGE_DIR}/val/${label} ]
    then
        mkdir -p ${IMAGE_DIR}/val/${label}
    fi

    # download images and split into train/val sets
    cd ${IMAGE_DIR}/train/${label}

    files=(${IMAGE_DIR}/train/${label}/*)
    if [ ${#files[@]} -gt 0 ];
    then
        echo "${IMAGE_DIR}/train/${label} already contains files, skipping"
        continnue
    fi

    echo "Downloading images"
    echo ${FPATHS_DIR}/${label}
    sudo cat ${FPATHS_DIR}/${label} | cut -d ',' -f 1 | parallel sudo curl -s -o ./#1-#2-#3-#4-#5

    cd ${ROOT_DIR}
    echo "Removing unopenable images"
    python src/tools/remove_unopenable_images.py ${IMAGE_DIR}/train/${label}

    echo "Splitting into train and val set"
    for f in $(ls -p ${IMAGE_DIR}/train/${label}/* | tail -n $(( $(ls -l ${IMAGE_DIR}/train/${label} | wc -l) / 10 )) )
    do
        sudo mv "$f" ${IMAGE_DIR}/val/${label}
    done
done