# This script is to visualize attention mask, output directory is under output
# DATASETS.TEST can have multiple test set, is source and target domain test sets, batch size is 1
# SOLVER.MAX_ITER determine how many image to visualize, if the number is greater than dataset size, just visualize entire dataset

#V2
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV2_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/city2foggy/model_0022799.pth" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val',)" SOLVER.MAX_ITER 999

#V1
python tools/train_net.py --config-file "configs/cityscapes2foggy/sapnetV1_R_50_C4.yaml" --visualize-attention-mask \
MODEL.WEIGHTS "pretrained/city2foggy-sapnet/model_0012199_MAP_46_60_SAPV1.pth" MODEL.DA_HEAD.NAME "SAPNet" \
DATASETS.TEST "('cityscapes_val','foggy-cityscapes_val',)" SOLVER.MAX_ITER 999
