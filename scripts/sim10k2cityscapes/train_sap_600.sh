source_model_path="pretrained/sim2city-baseline-source-only/model_600_37_85.pth"
# SAPNet 
weight_DA=0.1
size_train="(600,)"
size_test=600

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" --num-gpus 1 \
--setting-token "sim10k2city-sapnet-600-l${weight_DA}" \
INPUT.MIN_SIZE_TRAIN ${size_train} INPUT.MIN_SIZE_TEST ${size_test} \
MODEL.DA_HEAD.LOSS_WEIGHT ${weight_DA} \
MODEL.WEIGHTS ${source_model_path}
