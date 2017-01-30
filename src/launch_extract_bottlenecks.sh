#!/bin/bash

set -e
source ./core_slurm.sh

if [[ "$#" -ne 5 && "$#" -ne 4 && "$#" -ne 3  ]]; then
    echo "Must be run from /thesis root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_extract_bottlenecks.sh  runtime  memory  program_command [email]  [pwd]"
    echo "Example:"
    echo "./src/launch_extract_bottlenecks.sh 48:00:00 62GB \"python src/vision/extract_bottlenecks.py /scratch/network/dchouren/images/2015 /tiger/dchouren/thesis/resources/test_2015 vgg16\"  dchouren@princeton.edu /tigress/dchouren/thesis"
fi

runtime="$1"
memory="$2"
program_command="$3"

email="$4"
THESIS="$5"
SRC=$THESIS/src

# slurm_header runtime mem program name [additional_sbatch_instr]
SLURM_OUT=$THESIS/slurm
mkdir -p $SLURM_OUT


jobs=()

# months=( "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" )
months=( "13" )

year=$(echo ${image_root_dir} | rev | cut -d'/' -f 1 | rev)

for month in "${months[@]}"
do
  im_sub_dir=${image_root_dir}/${month}

  job_name="extract_bottlenecks_${model}_${year}_${month}"

  slurm_header $runtime $memory $program_command ${SLURM_OUT}/${job_name}.out > $SLURM_OUT/${job_name}.slurm

  jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))

  notify_email $SLURM_OUT/${job_name}.slurm > /tmp/$USER/${job_name}
done


jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








