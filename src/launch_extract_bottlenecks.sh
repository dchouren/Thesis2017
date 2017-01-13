#!/bin/bash

set -e

if [[ "$#" -ne 8 && "$#" -ne 7 && "$#" -ne 6  ]]; then
    echo "Must be run from /thesis/ root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_extract_bottlenecks.sh  image_root_dir  output_file  model  [email]  [pwd]"
    echo "Example:"
    echo "./src/launch_extract_bottlenecks $IMAGES/2015 $RESOURCES/test_2015.npy vgg16  dchouren@princeton.edu /tigress/dchouren/thesis"
fi


image_root_dir=$(readlink -f "$1")
output=$(readlink -f "$2")
model=$(readlink -f "$3")
email="$4"
THESIS="$5"
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

jobs=()

slurm_header "03:00:00" "8GB" "/bin/bash -c \"
    set -e
    python src/vision/extract_bottlenecks.py $image_root_dir $output $model
  \"" "extract_bottlenecks${image_root_dir}_${model}" > $SLURM_OUT/$extract_bottlenecks${image_root_dir}_${model}.slurm

  jobs+=($(sbatch $SLURM_OUT/$meeting_base.slurm | cut -f4 -d' '))
  notify_email $SLURM_OUT/$extract_bottlenecks${image_root_dir}_${model}.slurm > /tmp/$USER/$meeting_base
done


jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








