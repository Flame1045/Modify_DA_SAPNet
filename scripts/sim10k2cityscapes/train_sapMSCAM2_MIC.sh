#! /bin/bash
source_model_path="pretrained/sim2city-baseline-source-only/model_0033999_MAP_38_86.pth"
# source_model_path="pretrained/sim2city-sapnetv2/model_0015999_MAP_52.pth"
# SAPNetV2
weight_DA=0.1
weight_entropy=0.8
weight_diversity=0.3
weight_MIC_RPN_CLS=0.025
weight_MIC_RPN_LOC=0.2
weight_MIC_CLS=0.3
weight_MIC_BOX_REG=2.3
echo weight_MIC: ${weight_MIC}

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetMSCAM2_R_50_C4_MIC.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnetV2-l${weight_DA}-e${weight_entropy}-d${weight_diversity}" \
MODEL.DA_HEAD.MIC_RPN_CLS_WEIGHT ${weight_MIC_RPN_CLS} MODEL.DA_HEAD.MIC_RPN_LOC_WEIGHT ${weight_MIC_RPN_LOC} \
MODEL.DA_HEAD.MIC_CLS_WEIGHT ${weight_MIC_CLS} MODEL.DA_HEAD.MIC_BOX_REG_WEIGHT ${weight_MIC_BOX_REG} \
MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${weight_entropy} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${weight_diversity} \
MODEL.WEIGHTS ${source_model_path} 
