#!/bin/bash

set -e

# Notify_email description
function notify_email() {
  slurm_header "00:05:00" "100mb" "echo done" $1 "
#SBATCH --mail-type=end
#SBATCH --mail-user=$email
  "
}

## Generate a GPU slurm header.
#
# $1 = desired time
# $2 = desired memory
# $3 = command
# $4 = job name
# $5 = slurm output
function gpu_slurm_header() {
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
srun /usr/bin/time -f '%E elapsed, %U user, %S system, %M memory, %x status' /bin/bash -c \"set -e $3\"" 
}


## Generate a slurm header.
#
# $1 = desired time
# $2 = desired memory
# $3 = command
# $4 = job name
# $5 = slurm output
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
# Set working directory:
#SBATCH --workdir=$SLURM_OUT
$5
srun /usr/bin/time -f '%E elapsed, %U user, %S system, %M memory, %x status' /bin/bash -c \"set -e $3\""
}






