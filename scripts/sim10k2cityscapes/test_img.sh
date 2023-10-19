#v1
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 \
MODEL.WEIGHTS "pretrained/sim2city-sapnetv1/model_0005199_MAP_50_64.pth"

#v2
# python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 \
# MODEL.WEIGHTS "pretrained/sim2city-sapnetv2/model_0006199_MAP_50_34.pth" 
