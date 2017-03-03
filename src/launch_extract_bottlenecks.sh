#!/bin/bash

set -e
source $(dirname $0)/core_slurm.sh

if [[ "$#" -ne 4 ]]; then
    echo "Must be run from /thesis root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_extract_bottlenecks.sh  runtime  memory  model  year"
    echo "Example:"
    echo "./src/launch_extract_bottlenecks.sh 24:00:00 62GB vgg16 2015"
fi

runtime="$1"
memory="$2"
model="$3"
year="$4"

THESIS="/tigress/dchouren/thesis"
SRC=$THESIS/src

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=$THESIS/slurm
mkdir -p $SLURM_OUT


jobs=()

months=( "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" )

extract_bottlenecks_command="python /tigress/dchouren/thesis/src/vision/extract_bottlenecks.py"
base_arg_1="/scratch/network/dchouren/images/${year}"  # image dir
base_arg_2="/tigress/dchouren/thesis/resources/bottlenecks/${model}/${year}"  # output base
base_arg_3=$model

mkdir -p $base_arg_2

base_job_name="extract_bottlenecks"

for month in "${months[@]}"
do
  job_name="${base_job_name}_${model}_${year}_${month}"

  program_command="${extract_bottlenecks_command} ${base_arg_1}/${month} ${base_arg_2}/${month} ${base_arg_3}"

  echo $program_command

  gpu_slurm_header $runtime $memory "${program_command}" ${job_name} > $SLURM_OUT/${job_name}.slurm

  jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))
done


jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"

watch -n 0.2 squeue -u dchouren






