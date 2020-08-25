#!/bin/bash

# lr=0.01
#lr=0.1
lr=0.001
steps=10
max_norm=0.03
data=cifar10
#data=stl10
#data=mnist
#data=restricted_imagenet
#data=tiny_imagenet
root=data
#root=tiny_imagenet
cp=0.1
model=wide_resnet_trades,wide_resnet,wide_resnet
#model=wide_resnet,wide_resnet,wide_resnet
#model=aaron
#model=vgg
#model=resnet
#model=resnet34
#model=small_cnn
version=2
model_out=./checkpoint/${data}_${model}_v${version}_mix_mix_last_10
echo "model_out: " ${model_out}
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./main_adv_cus.py \
                        --lr ${lr} \
                        --data ${data} \
                        --model ${model} \
                        --step ${steps} \
                        --max_norm ${max_norm} \
                        --root ${root} \
                        --model_out ${model_out}.pth \
                        --corruption_prob ${cp} \
                        --train_sampler true \
                        --version ${version} \
                        --resuming 1 \
                        --resuming_model_dir partial_ensemble_PGD_trained_model_logits_summed_epoch_50 \
                        --model_dir ../model_cifar_wrn.pt,./generated_models/PGD_model_80
