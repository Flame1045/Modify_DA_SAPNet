#!/bin/bash
# echo SOURCE ONLY TRAINING
# bash scripts/sim10k2cityscapes/train_source_only.sh 
## choice ~38AP model

echo UDA FINETUNE TRAINING ON TARGET FOR PSEUDO IMAGE GENERATE
bash scripts/sim10k2cityscapes/sap_finetune_for_pseudo.sh 

# echo PSUEDO IMAGE DATASET GENERATE 
# bash scripts/sim10k2cityscapes/gen_pseudo.sh 

# echo PSUEDO IMAGE DATASET VISUALIZE 
# bash scripts/sim10k2cityscapes/visualize.sh

# echo SAPNET BASELINE TRAINING
# bash scripts/sim10k2cityscapes/train_sap.sh

# echo SAPNET LA-MIXUP TRAINING
# bash scripts/sim10k2cityscapes/train_sap_pseudo.sh 
