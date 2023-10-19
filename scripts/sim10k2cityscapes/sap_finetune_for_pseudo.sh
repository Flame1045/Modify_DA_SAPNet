#!/bin/bash
source_model_path="pretrained/sim2city-baseline-source-only/model_0001999_MAP38_83.pth"
# SAPNet 
weight_DA=0.1

Seeds=(35)
Source_weight=(0.1 0.3 0.7)
Target_weight=(0.1 0.3 0.7)
for i in "${Seeds[@]}";do
    for j in "${Source_weight[@]}";do
        for k in "${Target_weight[@]}";do
            echo "$i" "$j" "$k"
            python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4_Finetune.yaml" --num-gpus 1 \
            --setting-token "sim10k2city-sapnet-l${weight_DA}" \
            MODEL.FINETUNE_PSEUDO_SOURCE_WEIGHTS "$j" MODEL.FINETUNE_PSEUDO_TARGET_WEIGHTS "$k" \
            MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} MODEL.WEIGHTS ${source_model_path} SEED "$i"
        done
    done
done


