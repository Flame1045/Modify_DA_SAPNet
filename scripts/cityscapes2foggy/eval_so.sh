# evaluate source only object detector
echo "start to evaluate source only object detector"
model_path="pretrained/cityscapes-baseline/model_0019999_MAP_32_21_T002_V002.pth"
python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}


