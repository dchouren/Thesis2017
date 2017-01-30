#!/bin/bash

set -e
source $(dirname $0)/core_slurm.sh

if [[ "$#" -ne 7 && "$#" -ne 6 && "$#" -ne 5 ]]; then
    echo "Must be run from /thesis root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/generate_slurm.sh  runtime  memory  program_command  job_name  use_gpu"
    echo "Example:"
    echo "./src/generate_slurm.sh 48:00:00 62GB \"python src/vision/extract_bottlenecks.py /scratch/network/dchouren/images/2015/14/train /tiger/dchouren/thesis/resources/test_2015 vgg16\" test_2015_14 true dchouren@princeton.edu /tigress/dchouren/thesis"
fi

runtime="$1"
memory="$2"
program_command="$3"
job_name="$4"
use_gpu="$5"

email="$6"
THESIS="$7"
SRC=$THESIS/src

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=$THESIS/slurm
mkdir -p $SLURM_OUT

echo $SLURM_OUT

jobs=()

if [ "$use_gpu" = true ]; then
  echo "Using GPU"
  function_call=gpu_slurm_header
else
  echo "Using CPU"
  function_call=slurm_header
fi
echo $runtime
$function_call $runtime $memory "$program_command" ${SLURM_OUT}/${job_name}.out > $SLURM_OUT/${job_name}.slurm

jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))

notify_email $SLURM_OUT/${job_name}.slurm > /tmp/$USER/${job_name}

jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








