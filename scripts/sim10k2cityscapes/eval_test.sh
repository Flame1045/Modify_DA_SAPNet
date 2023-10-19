#!/bin/bash
SEED=35
# evaluate BASELINE
echo "start to evaluate BASELINE"
model_path=""

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_R_50_C4.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path} SEED ${SEED}

# evaluate Porposed
echo "start to evaluate Porposed"
model_path=""

python tools/train_net.py --config-file "configs/sim10k2cityscapes/sapnetV1_pseudo_dataset.yaml" --num-gpus 1 --eval-only \
MODEL.WEIGHTS ${model_path} SEED ${SEED}

