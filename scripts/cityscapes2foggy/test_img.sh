python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 \
MODEL.WEIGHTS "pretrained/model_0047999.pth" DATASETS.TEST "('foggy-cityscapes_val',)"

# python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --test-images MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.7 \
# MODEL.WEIGHTS "pretrained/cityscapes-baseline/model_0034999.pth" DATASETS.TEST "('foggy-cityscapes_val',)"

