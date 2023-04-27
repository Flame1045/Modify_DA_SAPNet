# echo "Training SAPNet iAFF pad zero"
# bash scripts/sim10k2cityscapes/train_sapiAFFpz.sh

# echo "Training SAPNet iAFF channel max pooling"
# bash scripts/sim10k2cityscapes/train_sapiAFFpool.sh

# echo "Training SAPNet MSCAM2 with MIC loss 0.1"
# bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC01.sh

# echo "Training SAPNet MSCAM2 with MIC loss 0.07"
# bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC007.sh

# echo "Training SAPNet MSCAM2 with MIC loss 0.05"
# bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC005.sh

# echo "Training SAPNet MSCAM2 with MIC loss 0.03"
# bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC003.sh

# echo "Training SAPNet MSCAM2 with MIC loss 0.01"
# bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC001.sh

# for i in {1..10}
# do
#     bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC001.sh "$i"
# done

# for fp in $(seq 0.001 .001 1.0)
# do
#     bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC001.sh $fp
# done

# for x in {100..1}; do
#      y=`bc <<< "scale=5; $x/100000"`
#      bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC.sh $y
# done

for x in {1..100}; do
     echo Number $x test
     bash scripts/sim10k2cityscapes/train_sapMSCAM2_MIC.sh 
done