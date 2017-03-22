





base_job_name="make_pairs"
sample_size="2000"

years=( "2006" "2007" "2008" "2009" "2010" "2011" "2012" "2013" "2014" "2015" )
months = months=( "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" )

for year in "${years[@]}"
do
    for month in "${months[@]}"
    do
        job_name="${year}_${month}_${base_job_name}"

        program_command="python /tigress/dchouren/thesis/src/vision/create_training_data.py ${year} ${month} ${sample_size}"

        echo $program_command

        gpu_slurm_header $runtime $memory "${program_command}" ${job_name} > $SLURM_OUT/${job_name}.slurm

        jobs+=($(sbatch $SLURM_OUT/${job_name}.slurm | cut -f4 -d' '))
    done
done


