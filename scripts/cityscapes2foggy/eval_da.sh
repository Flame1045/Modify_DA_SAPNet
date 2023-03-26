# evaluate object detector with DA
echo "start to evaluate object detector with DA"
model_path="pretrained/city2foggy/model_0020599_mAP_49_4995.pth"
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path}
