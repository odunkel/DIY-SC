#!/bin/bash

echo "Compute SAM masks"


SPAIR_DIR=./data/SPair-71k/JPEGImages

python preprocess_mask_sam.py $SPAIR_DIR
