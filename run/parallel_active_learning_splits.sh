#!/bin/bash

# This script is not recommended for parallelized computation
# Preferrable to use your distributed computing system

# Perform the SAIA experiment for 10 splits on 4 GPUs

cd ../experiments/ 

for i in {0..9}
do  
    export CUDA_VISIBLE_DEVICES=$(($i % 4))
    python active_learning.py --model HHVAEM --dataset boston --method mi --split $i & 
done
