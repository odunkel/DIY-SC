#!/bin/bash

echo "Preprocessing features"

SPAIR_DIR="./data/SPair-71k/JPEGImages"

for arg in sd dino sph; do [[ "$@" =~ "$arg" ]] && declare "$arg=true" || declare "$arg=false"; done

echo "Using arguments: sd=$sd, dino=$dino, sph=$sph"


if [[ "$sd" == true ]]; then
    echo "Computing SD features"
    python preprocess_map.py --base_dir $SPAIR_DIR --sd
elif [[ "$dino" == true ]]; then
    echo "Computing DINO features"
    python preprocess_map.py --base_dir $SPAIR_DIR --dino
elif [[ "$sph" == true ]]; then
    echo "Computing SphMap features"
    python preprocess_sph_feats.py --spair_dir $SPAIR_DIR --ckpt_path ./ckpts/sph_map/spair71k/ckpts/200.pth
fi
