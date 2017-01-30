#!/bin/bash

set -e
source ./core_slurm.sh

if [[ "$#" -ne 6 && "$#" -ne 5 && "$#" -ne 4  ]]; then
    echo "Must be run from /thesis root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_extract_bottlenecks.sh  runtime  memory  program_command  job_name"
    echo "Example:"
    echo "./src/launch_extract_bottlenecks.sh 48:00:00 62GB \"python src/vision/extract_bottlenecks.py /scratch/network/dchouren/images/2015/14/train /tiger/dchouren/thesis/resources/test_2015 vgg16\" test_2015_14 dchouren@princeton.edu /tigress/dchouren/thesis"
fi

runtime="$1"
memory="$2"
program_command="$3"
job_name="$4"

email="$5"
THESIS="$6"
SRC=$THESIS/src

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=$THESIS/slurm
mkdir -p $SLURM_OUT


jobs=()

slurm_header $runtime $memory $program_command ${SLURM_OUT}/${job_name}.out > $SLURM_OUT/${job_name}.slurm

jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))

notify_email $SLURM_OUT/${job_name}.slurm > /tmp/$USER/${job_name}

jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








