#!/bin/bash
pseudo_model_gen_path="pretrained/sim2city-sapnetv1/model_0006999_MAP_51_14.pth" # Source weight 0.1 target weight 0.3
# SAPNet 
weight_DA=0.1

# ##Generate pseudo image
python3 tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_Pseudo_gen.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnet-Pseudo_gen" \
MODEL.PSEUDO_WEIGHTS ${pseudo_model_gen_path}


