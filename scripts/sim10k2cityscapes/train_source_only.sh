#!/bin/bash

Seeds=(35 35 35)
for i in "${Seeds[@]}";do
    echo "$i"
    python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 \
    --setting-token "sim10k2city_source_only" SEED "$i"
done
