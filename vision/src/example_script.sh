#!/usr/bin/env bash
# not tested

GEOID=187147
PLACETYPEID=10022
CONDA_ENV="daneel"

TASK=thumbnail
DATASET=dslr

IMAGE_DIR=/spindle/${USER}/images/${TASK}/${DATASET}

ROOT_DIR=/home/${USER}/vision

uc_adhoc

PATHS_DIR=${ROOT_DIR}/resources/paths/${TASK}/${DATASET}
FPATHS_DIR=${ROOT_DIR}/resources/fpaths/${TASK}/${DATASET}
PATH_FILENAME=$1


# pull path, mediaid, locationid information
hive -e <<EOF
SELECT
    m.path,
    m.id as mediaid,
    j.id as locationid,
    m.memberid,
    m.uploaded_date
FROM
    (SELECT
        l.id
    FROM
        t_locationpaths lp
    JOIN
        t_location l
    ON l.id=lp.locationid
    WHERE
        lp.parentid=${GEOID}
        AND l.placetypeid=${PLACETYPEID}
    ) j
JOIN
    t_media_locations ml
ON j.id=ml.locationid
JOIN
    t_media m
ON ml.mediaid=m.id
WHERE m.status=4
ORDER BY
    locationid DESC
EOF
> ${ROOT_DIR}/resources/paths/${TASK}/${DATASET}/${PATH_FILENAME}

source activate ${CONDA_ENV}

for label in "${LABELS}"; do
    echo $label
    # format paths for image downloading
    python src/tools/format_image_paths.py ${PATHS_DIR} ${FPATHS_DIR} '${label}' -1

    if [ -ne ${IMAGE_DIR}/train/${label} ]; do
        mkdir -p ${IMAGE_DIR}/train/${label}
    done
    if [ -ne ${IMAGE_DIR}/val/${label} ]; do
        mkdir -p ${IMAGE_DIR}/val/${label}
    done

    # download images and split into train/val sets
    cd ${IMAGE_DIR}/train
    echo "Downloading images"
    sudo cat ${ROOT_DIR}/${FPATH_DIR}/${PATH_FILENAME} | cut -d ',' -f 1 | sudo parallel sudo curl -s -o ./#1-#2-#3-#4-#5

    cd ${ROOT_DIR}
    echo "Removing unopenable images"
    python src/tools/remove_unopenable_images ${IMAGE_DIR}/train

    python src/vision/augment_images.py   # only used this for the menu classifier, but if oyu need to expand your dataset you can apply shears and other transforms

    echo "Splitting into train and val set"
    VAL_PCT=0.1
    for f in $(ls -p ${IMAGE_DIR}/train/* | tail -n $(( $(ls -l milan | wc -l) * ${VAL_PCT} ));
        do sudo mv ${IMAGE_DIR}/train/"$f" ${IMAGE_DIR}/val/
    done
done

cd ${ROOT_DIR}
MODEL=resnet50
NUM_EPOCHS=100
BATCH_SIZE=256
LABEL=dslr
python src/vision/bottleneck_train.py ${MODEL} ${TASK} ${DATASET} ${NUM_EPOCHS} ${BATCH_SIZE} ${LABEL}

python src/vision/extract_bottlenecks.py   # save bottleneck representation for faster predicting

python src/vision/test_model_test.py $RESOURCES/thumbnail/dslr/trained_models/finetuned_vgg16_200_13926.75_dslr.h5 resources/fpaths/thumbnail/milan $IMAGES/testsets/milan output/thumbnail/dslr/finetuned_menu_face_dslr_milan dslr good # save predictions to output/

python src/html/examine_keras_output.py  # build html pages




echo "Done"