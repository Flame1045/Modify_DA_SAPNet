size_train="(600,)"
size_test=600

python tools/train_net.py --config-file "configs/sim10k2cityscapes/source_only_R_50_C4.yaml" --num-gpus 1 --setting-token "sim10k2city_source_only" \
INPUT.MIN_SIZE_TRAIN ${size_train}  INPUT.MIN_SIZE_TEST ${size_test} \