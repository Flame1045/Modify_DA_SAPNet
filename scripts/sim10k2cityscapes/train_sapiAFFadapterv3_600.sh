source_model_path="pretrained/sim2city-baseline-source-only/model_600.pth"
# source_model_path="pretrained/sim2city-sapnetv2/model_0015999_MAP_52.pth"
# SAPNetV2
weight_DA=0.1
weight_entropy=0.8
weight_diversity=0.3
size_train="(600,)"
size_test=600

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetiAFFadapter_R_50_C4v3.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnetV2-l${weight_DA}-e${weight_entropy}-d${weight_diversity}" \
INPUT.MIN_SIZE_TRAIN ${size_train} INPUT.MIN_SIZE_TEST ${size_test} \
MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${weight_entropy} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${weight_diversity} \
MODEL.WEIGHTS ${source_model_path} 
