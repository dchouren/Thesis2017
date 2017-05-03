model=$1

./src/generate_slurm.sh 1:00:00 62GB "python /tigress/dchouren/thesis/src/vision/predict_siamese.py resnet50 nadam ${model}" predict_${model} true /tigress/dchouren/thesis/out