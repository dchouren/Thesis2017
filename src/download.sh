# ./download.sh /tigress/dchouren/thesis/resources/paths/2013 /scratch/dchouren/images/2013 1

PATH_DIR=$1
IMAGE_DIR=$2
NUM_P=$3

cd $PATH_DIR 
for f in *
do
  if [ -d "${IMAGE_DIR}/${f}" ]; then
    continue
  fi
  
  echo "$f"
  mkdir ${IMAGE_DIR}/"$f"
  cd ${IMAGE_DIR}/"$f"
  #cat ${PATH_DIR}/"$f" | cut -d ',' -f 1 | xargs -P $NUM_P -n 1 curl -s -O 
  cat ${PATH_DIR}/"$f" | cut -d ',' -f 1 | parallel curl -L -s -O
done 
