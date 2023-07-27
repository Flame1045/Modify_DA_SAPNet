source_model_path="pretrained/cityscapes-baseline/model_0019999_MAP_32_21_T002_V002.pth"
# SAPNet + MSCAM
weight_DA=0.6

python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --num-gpus 1 \
--setting-token "city2foggy-sapnet-mscam-l${weight_DA}" MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.WEIGHTS ${source_model_path}
