# for x in {1..5}; do
#      echo Number $x test baseline
#      bash scripts/sim10k2cityscapes/train_sap_600.sh 
# done

# for x in {1..5}; do
#      echo Number $x test adapterv2
#      bash scripts/sim10k2cityscapes/train_sapiAFFadapterv2.sh 
# done

# for x in {1..4}; do
#      echo Number $x test basline standard size
#      bash scripts/sim10k2cityscapes/train_sap_600.sh 
# done

for x in {1..5}; do
     echo Number $x test MSCAM standard size
     bash scripts/sim10k2cityscapes/train_sapMSCAM_600.sh 
done

for x in {1..5}; do
     echo Number $x test adapterv2 standard size
     bash scripts/sim10k2cityscapes/train_sapiAFFadapterv2_600.sh 
done

# for x in {1..5}; do
#      echo Number $x test baseline full size
#      bash scripts/sim10k2cityscapes/train_sap.sh 
# done




