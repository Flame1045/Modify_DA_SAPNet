# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector

#v1
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 SOLVER.MAX_ITER 500  \
MODEL.WEIGHTS "pretrained/sim2city-sapnetv1/model_0005199_MAP_50_64.pth"

#v2
python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 SOLVER.MAX_ITER 500 \
MODEL.WEIGHTS "pretrained/sim2city-sapnetv2/model_0006199_MAP_50_34.pth" 
