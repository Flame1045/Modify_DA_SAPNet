# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector

#V2
python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 SOLVER.MAX_ITER 500 \
MODEL.WEIGHTS "pretrained/city2foggy/model_0020599_mAP_49_4995.pth"

#V1
python tools/train_net.py --config-file "configs/cityscapes2foggy/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.85 SOLVER.MAX_ITER 500 \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0017799_MAP_48_72.pth"
