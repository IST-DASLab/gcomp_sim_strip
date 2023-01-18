#!/usr/bin/env bash

NUM_NODES=${1:-2}
batch_size=$(( 512 / $NUM_NODES ))

# For CIFAR100 change batch size and dataset dir parameter
#batch_size=$(( 128 / $NUM_NODES ))

log_dir="cifar10_test"
mkdir $log_dir

#python -m torch.distributed.launch --nproc_per_node=$NUM_NODES \

python -m torch.distributed.launch --nproc_per_node=4 \
cifar_train.py --epochs 200 --dataset-dir ~/Datasets/cifar10 \
--batch-size $batch_size --log-dir $log_dir 2>&1 | tee $log_dir/out