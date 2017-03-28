#!/bin/bash

set -e
source $(dirname $0)/core_slurm.sh


THESIS="/tigress/dchouren/thesis"
SLURM=$THESIS/slurm
SLURM_OUT=$THESIS/out

runtime=01:30:00
memory=32GB

model="test.model"
year="2015"
month="01"
n_iter="1000"


base_job_name="tsne"

perplexities=( "10" "20" "30" "40" "50" "60" "70" "80" "90" "100" "120" "140" "160" "180" "200" "250" "300" "400" "500" "600" "700" "800" "900" "1000" )
for perplexity in "${perplexities[@]}"
do
    job_name="${perplexity}_${base_job_name}"

    program_command="python /tigress/dchouren/thesis/src/vision/tsne.py ${model} ${year} ${month} ${perplexity} ${n_iter}"

    echo $program_command

    slurm_header $runtime $memory "$program_command" ${job_name} > $SLURM/${job_name}.slurm

    jobs+=($(sbatch $SLURM/${job_name}.slurm | cut -f4 -d' '))
done


watch -n 0.2 squeue -u dchouren


