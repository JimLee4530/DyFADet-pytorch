##!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/anet_tsp_clo.yaml --output pretrainedv3
echo "start testing..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/anet_tsp_clo.yaml ckpt/anet_tsp_clo_pretrainedv3/epoch_014.pth.tar
