#!/bin/bash

model=wide_resnet
#model=small_cnn
# model=vgg
#model=resnet
#model=resnet18
#defense=plain
defense=trades
data=cifar10
#data=stl10
#data=mnist
#data=restricted_imagenet
# data=tiny_imagenet
root=data
n_ensemble=1
steps=( 100 )
#steps=( 500 )
attack=Linf
#attack=L2
#attack=CW
#max_norm=0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.065,0.07
num_start=1
# max_norm=0.03137
max_norm=0.03
for k in "${steps[@]}"
do
    echo "running" $k "steps"
    CUDA_VISIBLE_DEVICES=0 python acc_under_attack.py \
        --model $model \
        --defense $defense \
        --data $data \
        --root $root \
        --n_ensemble $n_ensemble \
        --steps $k \
        --max_norm $max_norm \
        --attack $attack \
        --alpha 2 \
        --num_start $num_start \
	    --model_dir /home/chengminhao/github_v/CAT-Customized-Adversarial-Training-for-Improved-Robustness/checkpoint/cifar10_wide_resnet_v2_ce_ce_last_10.pth79

done
