##!/bin/bash

echo "start training"
CUDA_VISIBLE_DEVICES=$1 python trainv2.py ./configs/anet_tspv4.yaml --output pretrained