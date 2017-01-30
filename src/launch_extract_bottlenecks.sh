#!/bin/bash

set -e
source ./core_slurm.sh

if [[ "$#" -ne 5 && "$#" -ne 4 && "$#" -ne 3  ]]; then
    echo "Must be run from /thesis/ root dir NOT from /src if pwd is not specified"
    echo "Usage:"
    echo "./src/launch_extract_bottlenecks.sh  image_root_dir  output_file  model  [email]  [pwd]"
    echo "Example:"
    echo "./src/launch_extract_bottlenecks.sh /scratch/network/dchouren/images/2015 /tiger/dchouren/thesis/resources/test_2015 vgg16  dchouren@princeton.edu /tigress/dchouren/thesis"
fi


image_root_dir=$1
output=$2
model=$3
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

# months=( "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" )
months=( "13" )

year=$(echo ${image_root_dir} | rev | cut -d'/' -f 1 | rev)

for month in "${months[@]}"
do
  im_sub_dir=${image_root_dir}/${month}

  job_name="extract_bottlenecks_${model}_${year}_${month}"

  slurm_header "36:00:00" "32GB" "/bin/bash -c \"
      set -e
      python ${SRC}/vision/extract_bottlenecks.py $im_sub_dir $output $model
    \"" ${SLURM_OUT}/${job_name}.out > $SLURM_OUT/${job_name}.slurm

  jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))

  notify_email $SLURM_OUT/${job_name}.slurm > /tmp/$USER/${job_name}
done


jobs=$(echo ${jobs[@]} | tr ' ' ':')
echo "SLURM JOBS" $jobs


echo
echo "************************************************************"
echo "DONE WITH EVERYTHING"
echo "************************************************************"








