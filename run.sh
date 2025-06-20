#!/bin/bash

# gpu param need to be specified: $1
gpu=$1

# integers=(5012 3286 2824 9935 2424)
integers=(5012 3286 2679 9935 2424)

for integer in "${integers[@]}"
do
    echo "Starting run experiment on seed $integer"
     python train.py --data_path Pavia/ --source_domain PaviaU --target_domain PaviaC --training_sample_ratio 0.5 --re_ratio 1 --seed $integer --gpu $gpu
#     python train.py --data_path Houston/ --source_domain Houston13 --target_domain Houston18 --training_sample_ratio 0.8 --re_ratio 5 --flip_augmentation --radiation_augmentation --seed $integer --gpu $gpu --embed 256
#    python train-center.py --data_path HyRANK/ --source_domain Dioni --target_domain Loukia --training_sample_ratio 0.8 --re_ratio 1 --seed $integer --gpu $gpu --lambda_1 10.0 --lambda_2 10.0 --embed 512
    echo "Finished run experiment on seed $integer"
    echo "-------------------------"
done