#!/bin/bash

set -e

if [[ "$#" -ne 8 && "$#" -ne 7 && "$#" -ne 6  ]]; then
    echo "Must be run from /thesis/ root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_all.sh  image_root_dir  labels_dir  output_dir  model_type  num_epochs  batch_size  [email]  [pwd]"
    echo "Example:"
    echo "./src/launch_all.sh /tigress/dchouren/thesis/filtered_features/250/gray_cropped_combined/  /tigress/dchouren/thesis/filtered_labels/words/250/  /tigress/dchouren/thesis/lasagne_output/  cnn  10  100  dchouren@princeton.edu /tigress/dchouren/thesis"
fi


feats=$(readlink -f "$1")
labels=$(readlink -f "$2")
output=$(readlink -f "$3")
model_type="$4"
num_epochs="$5"
batch_size="$6"
email="$7"
THESIS="$8"
SRC=$THESIS/src

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=$THESIS/slurm
mkdir -p $SLURM_OUT

# notify_email description
function notify_email() {
  slurm_header "00:05:00" "100mb" "echo done" $1 "
#SBATCH --mail-type=end
#SBATCH --mail-user=$email
  "
}

function slurm_header() {
echo "#!/bin/sh
# Request runtime
#SBATCH --time=$1
# Request a number of CPUs per task:
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
# Request a number of nodes:
#SBATCH --nodes=1
# Request an amount of memory per node:
#SBATCH --mem=$2
# Specify a job name:
#SBATCH -J $4
#SBATCH -o $4
# Request GPU stuff
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1 
#SBATCH --gres=gpu:1
# Set working directory:
#SBATCH --workdir=$SLURM_OUT
$5
srun /usr/bin/time -f '%E elapsed, %U user, %S system, %M memory, %x status' $3"
}


#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1 
#SBATCH --gres=gpu:1


meeting_bases=("ES2014a" "ES2014b" "ES2014c" "ES2014d" "ES2015a" "ES2015b" "ES2015c" "ES2015d" "ES2016a" "ES2016b" "ES2016c" "ES2016d")

all_lasagne=()

# Lasagne Slurm Creation
for meeting_base in ${meeting_bases[@]}; do
  echo $meeting_base
  slurm_header "05:00:00" "8GB" "/bin/bash -c \"
    set -e
    cd $SRC
    python lasagne_base.py  $feats  $labels  $output  $model_type  $num_epochs  $batch_size  0  $meeting_base 
  \"" "$meeting_base$num_epochs$model_type" > $SLURM_OUT/$meeting_base.slurm

  all_lasagne+=($(sbatch $SLURM_OUT/$meeting_base.slurm | cut -f4 -d' '))
  notify_email $meeting_base > /tmp/$USER/$meeting_base
done

all_lasagne=$(echo ${all_lasagne[@]} | tr ' ' ':')
echo "SLURM JOBS" $all_lasagne


# while [ ! -f $SLURM_OUT/lasagne ]; do
#   sleep 5
# done

echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








