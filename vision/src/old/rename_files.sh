#!/usr/bin/env bash

DIR=$1
TMP=tmp
TMP2=tmp2

SIZE=224

pushd ${DIR}

if [ ! -d "$TMP" ]; then
    mkdir $TMP
fi



declare -a classes=("menu" "receipt" "other")

# augmenting images
for CLASS in "${classes[@]}"
do
    echo "$CLASS"
    if [ ! -d "${CLASS}_${TMP}" ]; then
        mkdir ${CLASS}_${TMP}
    fi
    if [ ! -d "${CLASS}_${TMP2}" ]; then
        mkdir ${CLASS}_${TMP2}
    fi
    if [ ! -d "${CLASS}" ]; then
        mkdir ${CLASS}
    fi

    rm ${CLASS}_${TMP}/* || true
    rm ${CLASS}_${TMP2}/* || true

    echo "Resizing"
    a=1
    for i in true_${CLASS}/*; do
      new=$(printf "${CLASS}_%04d.jpg" "$a") #04 pad to length of 4
      convert "$i" -resize ${SIZE}x${SIZE}! ${CLASS}_${TMP}/"$new"
      let a=a+1
    done

    echo "Augmenting"
    python ../src/augment_images.py ${CLASS}_${TMP} ${CLASS}_${TMP2} $CLASS 5

    echo "Final naming"
    a=1
    for i in ${CLASS}_${TMP2}/*; do
      new=$(printf "${CLASS}_%04d.jpg" "$a") #04 pad to length of 4
      convert "$i" -resize ${SIZE}x${SIZE}! ${CLASS}/"$new"
      let a=a+1
    done
done

#TOTAL_NUM=$(ls other | wc -l)
#TRAIN_VAL_SPLIT=$[${TOTAL_NUM} / 4 * 3]

