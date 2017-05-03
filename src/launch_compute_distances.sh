#!/bin/bash

set -e
source $(dirname $0)/core_slurm.sh


THESIS="/tigress/dchouren/thesis"
SLURM=$THESIS/slurm
SLURM_OUT=$THESIS/out

runtime=03:00:00
memory=62GB


pair_files=( "2015_01_32000.h5" "2014_02_32000.h5" "2015_02_32000.h5" "2015_03_32000.h5" "middlebury_diffuser.h5" "2013-5_10m_tl.h5" "2014_03_32000.h5" "2015_04_32000.h5" "2013-5_10m_user.h5" "2014_04_32000.h5" "2015_05_32000.h5" "2013-5_1m.h5" "2014_05_32000.h5" "2015_06_32000.h5" "middlebury_sameuser.h5" "2014_06_32000.h5" "2015_07_32000.h5" "new_2013_2014_2015_all.h5" "2013-5_1m_tl.h5" "2014_07_32000.h5" "2015_08_32000.h5" "new_2015_all.h5" "2013-5_1m_user.h5" "2014_08_32000.h5" "2015_09_32000.h5" "2013-5_30m_user_2h.h5" "2014_09_32000.h5" "2015_10_32000.h5" "stereo_diffuser.h5" "2014_01_32000.h5" "2014_10_32000.h5" "2015_11_32000.h5" "stereo_sameuser.h5" "2014_11_32000.h5" "2015_12_32000.h5" "2014_12_32000.h5" )
# pair_files=( "2015_01_32000.h5" "2014_02_32000.h5" )

for pair_file in "${pair_files[@]}"; do
    job_name="bot_dist_${pair_file}"

    program_command="python /tigress/dchouren/thesis/src/vision/extract_bottlenecks.py ${pair_file}"

    echo $program_command

    gpu_slurm_header $runtime $memory "$program_command" ${job_name} > $SLURM/${job_name}.slurm

    jobs+=($(sbatch $SLURM/${job_name}.slurm | cut -f4 -d' '))
done



