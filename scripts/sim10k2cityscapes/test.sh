#!/bin/bash

# # Define the start value of the range with a negative increment value
# for i in $(seq 0.001 0.0001 0.0001)
# do
#      # Print the value with 'ID-'
#      echo "The value is $i"
# done

for x in {100..1}; do
     y=`bc <<< "scale=5; $x/100000"`
     echo $y
done
