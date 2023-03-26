source_model_path="pretrained/sim2city-baseline-source-only/model_0033999_MAP_38_86.pth"
# SAPNet 
weight_DA=0.1

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnet-cyclegan-l${weight_DA}" \
MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.WEIGHTS ${source_model_path}
