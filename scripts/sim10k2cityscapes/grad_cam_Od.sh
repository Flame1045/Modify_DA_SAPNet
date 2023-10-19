# This script is to visualize grad cam, output directory is under test_images

# grad cam is to track where location on image is object detector focus
# --grad-cam-object-detection is to visualize grad cam of object detector

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
# --grad-cam-object-detection \
# MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 SOLVER.MAX_ITER 500 \
# DATASETS.TEST "('cityscapes-car2_val',)" \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth"

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
--grad-cam-object-detection \
MODEL.ROI_HEADS.NAME "Res5ROIHeads_" MODEL.BACKBONE.NAME "build_resnet_backbone_" \
MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.75 SOLVER.MAX_ITER 500 \
MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth"

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" --visualize-attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth" \
# DATASETS.TEST "('sim10k_train', 'cityscapes-car2_val')" SOLVER.MAX_ITER 478