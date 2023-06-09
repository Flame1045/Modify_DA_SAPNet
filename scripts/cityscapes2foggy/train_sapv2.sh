source_model_path="pretrained/cityscapes-baseline-source-only/model_0019999_MAP_32_21_T002_V002.pth"

# SAPNetV2
weight_DA=0.6
weight_entropy=1.0
weight_diversity=0.2

python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --num-gpus 1 \
--setting-token "city2foggy-sapnetV2-l${weight_DA}-e${weight_entropy}-d${weight_diversity}" \
MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} MODEL.DA_HEAD.TARGET_ENT_LOSS_WEIGHT ${weight_entropy} MODEL.DA_HEAD.TARGET_DIV_LOSS_WEIGHT -${weight_diversity} \
MODEL.WEIGHTS ${source_model_path}

