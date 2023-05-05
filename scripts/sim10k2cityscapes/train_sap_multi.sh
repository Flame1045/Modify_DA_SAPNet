# echo "Training SAPNet iAFF pad zero"
# bash scripts/sim10k2cityscapes/train_sapiAFFpz.sh

# echo "Training SAPNet iAFF channel max pooling"
# weight_MIC_RPN_CLS=$1 # 0.025
# weight_MIC_RPN_LOC=$2 # 0.2
# weight_MIC_CLS=$3    # 0.3
# weight_MIC_BOX_REG=$4 # 2.3

# for rc in {26..1..5}; do
#     rcf=`bc <<< "scale=3; $rc/1000"`
#     for rl in {21..1..5}; do
#         rlf=`bc <<< "scale=2; $rl/100"`
#         for c in {31..1..5}; do
#             cf=`bc <<< "scale=2; $c/100"`
#             for b in {22..1..7}; do
#                 bf=`bc <<< "scale=1; $b/10"`
#                 echo $rcf $rlf $cf $bf
#                 bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC.sh $rcf $rlf $cf $bf
#             done
#         done
#     done
# done


for x in {1..5}; do
     echo Number $x test for v1
     bash scripts/sim10k2cityscapes/train_sapiAFFadapter.sh 
done

for x in {1..5}; do
     echo Number $x test for v2
     bash scripts/sim10k2cityscapes/train_sapiAFFadapterv2.sh 
done

for x in {1..5}; do
     echo Number $x test for v3
     bash scripts/sim10k2cityscapes/train_sapiAFFadapterv3.sh 
done