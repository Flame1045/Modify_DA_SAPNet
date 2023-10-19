
# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
# --grad-cam-source-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
# --grad-cam-source-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
# --grad-cam-target-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
# --grad-cam-target-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME0.1-23-09-27_20-26/model_0011999.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478


# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-source-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-l0.1-23-09-14_00-51/model_0001399.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-source-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-l0.1-23-09-14_00-51/model_0001399.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-target-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-l0.1-23-09-14_00-51/model_0001399.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-target-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-l0.1-23-09-14_00-51/model_0001399.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# >>>>>>>>>>>>>>>Baseline>>>>>>>>>>>>>>>
# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-source-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-baseline0.1-23-10-03_13-44/model_0006199.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-source-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-baseline0.1-23-10-03_13-44/model_0006199.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-target-doamin --backbone-feature \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-baseline0.1-23-10-03_13-44/model_0006199.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478

# python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" \
# --grad-cam-target-doamin --attention-mask \
# MODEL.WEIGHTS "/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-baseline0.1-23-10-03_13-44/model_0006199.pth" \
# DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
# SOLVER.MAX_ITER 478
# <<<<<<<<<<<<<<<Baseline<<<<<<<<<<<<<<<

# >>>>>>>>>>>>>>>ILLUME_a>>>>>>>>>>>>>>>
model_path="/media/ee4012/Disk3/Eric/Modify_DA_SAPNet/outputs/output-sim10k2city-sapnet-ILLUME-wodownsize0.1-23-10-08_17-00/model_0006999.pth"

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
--grad-cam-source-doamin --backbone-feature \
MODEL.WEIGHTS ${model_path} \
DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
SOLVER.MAX_ITER 478

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
--grad-cam-source-doamin --attention-mask \
MODEL.WEIGHTS ${model_path} \
DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
SOLVER.MAX_ITER 478

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
--grad-cam-target-doamin --backbone-feature \
MODEL.WEIGHTS ${model_path} \
DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
SOLVER.MAX_ITER 478

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnet_ILLUME.yaml" \
--grad-cam-target-doamin --attention-mask \
MODEL.WEIGHTS ${model_path} \
DATASETS.TEST "('sim10k_train','cityscapes-car2_val')"  MODEL.BACKBONE.NAME "build_resnet_backbone_" \
SOLVER.MAX_ITER 478
# <<<<<<<<<<<<<<<ILLUME_a<<<<<<<<<<<<<<<