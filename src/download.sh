# ./download.sh /tigress/dchouren/thesis/resources/paths/2013 /scratch/dchouren/images/2013 1

path_dir=$1
image_dir=$2
NUM_P=$3

cd $path_dir
for f in *
do
  echo "$f"
  mkdir ${image_dir}/"$f"
  cd ${image_dir}/"$f"
  cat ${path_dir}/"$f" | cut -d ',' -f 1 | xargs -P $NUM_P -n 1 curl -s -O 
done 
