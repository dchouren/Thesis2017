#!/bin/bash

set -e
source $(dirname $0)/core_slurm.sh


THESIS="/tigress/dchouren/thesis"
SLURM=$THESIS/slurm
SLURM_OUT=$THESIS/out

runtime=12:00:00
memory=62GB


base_job_name="make_pairs"
sample_size="3000"

years=( "2008" "2009" "2010" )
months=( "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" )
for year in "${years[@]}"
do
    for month in "${months[@]}"
    do
        job_name="${year}_${month}_${base_job_name}"

        program_command="python /tigress/dchouren/thesis/src/vision/create_training_data.py ${year} ${month} ${sample_size}"

        echo $program_command

        slurm_header $runtime $memory "$program_command" ${job_name} > $SLURM/${job_name}.slurm

        jobs+=($(sbatch $SLURM/${job_name}.slurm | cut -f4 -d' '))
    done
done


