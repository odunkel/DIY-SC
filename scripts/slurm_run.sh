#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 0-2:00:00
#SBATCH --gres gpu:1
#SBATCH -o/CT/adversarial_tester/work/sc/diysc/%j-slurm.out

start=`date +%s`

echo "Run DIY-SC"

cd /CT/adversarial_tester/work/sc/diysc

export WANDB_API_KEY=2433dc1c3dc43a1a020e50f8f318babd1d4fc6c6

# bash scripts/precompute_features.sh --dino
# bash scripts/compute_sam_masks.sh

# python pck_train.py --config configs/eval_spair.yaml --ONLY_DINO --LOAD ckpts/0300_dino_spair/best.pth

# bash scripts/precompute_features.sh --sph

# python gen_pseudo_labels.py --dataset_version v01 --filter_sph --subsample 300 --split val --spair_dir "/CT/datasets03/semantic_correspondence/SPair-71k/" --only_dino --sph_path_suffix "_sph_cleanimpl" --dino_path_suffix "_dino_implclean"
# python gen_pseudo_labels.py --dataset_version v01 --filter_sph --subsample 30000 --spair_dir "/CT/datasets03/semantic_correspondence/SPair-71k/" --only_dino --sph_path_suffix "_sph_cleanimpl" --dino_path_suffix "_dino_implclean"

# python pck_train.py --config configs/train_spair.yaml --EXP_ID 1 --TRAIN_DATASET_ANN_DIR '/CT/adversarial_tester/work/sc/diysc/data/pair_annotations/spair/v01' --ONLY_DINO
# python pck_train.py --config configs/train_spair.yaml --EXP_ID 2 --TRAIN_DATASET_ANN_DIR '/scratch/inf0/user/oduenkel/sc/data/spair/pair_annotations/v144' --ONLY_DINO
# python pck_train.py --config configs/train_spair.yaml --EXP_ID 3 --TRAIN_DATASET_ANN_DIR '/scratch/inf0/user/oduenkel/sc/data/spair/pair_annotations/v145' --ONLY_DINO

# SD+DINO
# python pck_train.py --config configs/eval_spair.yaml --LOAD ckpts/0280_spair/best.pth

# python gen_pseudo_labels.py --dataset_version v02 --filter_sph --subsample 300 --split val --spair_dir "/CT/datasets03/semantic_correspondence/SPair-71k/"
# python gen_pseudo_labels.py --dataset_version v02 --filter_sph --subsample 30000 --spair_dir "/CT/datasets03/semantic_correspondence/SPair-71k/"
# python pck_train.py --config configs/train_spair.yaml --EXP_ID 4 --TRAIN_DATASET_ANN_DIR '/CT/adversarial_tester/work/sc/diysc/data/pair_annotations/spair/v02'

python pck_train.py --config configs/eval_spair.yaml --LOAD results/exps/exp_0001_train_adpt/best.pth --ONLY_DINO

end=`date +%s`
runtime=$((end-start))
echo "Took in total $runtime s"

echo "Done"
