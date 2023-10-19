#!/bin/bash
source_model_path="pretrained/sim2city-baseline-source-only/model_0001999_MAP38_83.pth"
# SAPNet 
weight_DA=0.1
Seeds=(35 35 35)
for i in "${Seeds[@]}";do
    echo "$i"
    python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4_ORALCLE.yaml" --num-gpus 1 \
    --setting-token "sim10k2city-sapnet-baseline${weight_DA}" \
    MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} MODEL.WEIGHTS ${source_model_path} SEED "$i"
done
