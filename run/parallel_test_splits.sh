#!/bin/bash

#
# Copyright (c) 2022 by Ignacio Peis, UC3M.
# All rights reserved. This file is part of the HH-VAEM, and is released under
# the "MIT License Agreement". Please see the LICENSE file that should have
# been included as part of this package.
#

# This script is not recommended for parallelized computation
# Preferrable to use your distributed computing system

# Perform the SAIA experiment for 10 splits on 4 GPUs

cd ../ 

for i in {0..9}
do  
    export CUDA_VISIBLE_DEVICES=$(($i % 4))
    python test.py --model HHVAEM --dataset boston --split $i & 
done
