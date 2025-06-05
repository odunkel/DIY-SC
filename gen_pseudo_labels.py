import os
import math
import json
import random
import argparse
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from utils.utils_pseudo_gt import (
    resize, sample_spair_img_pairs, normalize_features, 
    get_cyclic_consistent_kps_for_pair, filter_out_with_sph, 
    sample_ordered_n_tupls, aggregate_spair_data_in_df
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class PairsDataset(Dataset):
    """
    Each sample in this dataset is an n-tuple (of image paths).
    For each sample, this class performs the following steps:
      1. Loads images, masks, and features and resizes them.
      2. Computes foreground keypoint indices from masks.
      3. Computes cyclic-consistent correspondences (along a chain).
      4. [Optionally] applies spherical filtering.
      5. Returns annotation dictionaries.
    """
    def __init__(self, pairs, img_size=840, edge_pad=False, num_patches=60,device='cuda',
                 pseudo_gt_gen_mode="nearest_neighbor", only_dino=False,
                 cyclic_consistency_mode="eucl_dist",cyclic_consistent=False, rej_th_patch=2, 
                 use_mask=True, n_tupl=4,filter_sph=False, filter_sph_threshold=0.15,
                 dino_path_suffix='_dino', sd_path_suffix='_sd', sph_path_suffix='_sph', mask_suffix='_mask'):
        super(PairsDataset, self).__init__()
        self.pairs = pairs
        self.img_size = img_size
        self.edge_pad = edge_pad
        self.num_patches = num_patches
        self.pseudo_gt_gen_mode = pseudo_gt_gen_mode
        self.only_dino = only_dino
        self.cyclic_consistency_mode = cyclic_consistency_mode
        self.filter_sph = filter_sph
        self.filter_sph_threshold = filter_sph_threshold
        self.device = device
        self.use_mask = use_mask
        self.cyclic_consistent = cyclic_consistent
        self.n_tupl = n_tupl
        self.rej_th_patch = rej_th_patch
        self.dino_path_suffix = dino_path_suffix
        self.sd_path_suffix = sd_path_suffix
        self.sph_path_suffix = sph_path_suffix
        self.mask_suffix = mask_suffix

        if self.pseudo_gt_gen_mode == "nearest_neighbor": self.n_tupl = 2

        self.scale_factor = self.img_size / self.num_patches
        coords = torch.stack(torch.meshgrid(torch.arange(self.num_patches), torch.arange(self.num_patches), indexing="ij"), dim=-1).reshape(-1, 2)
        self.kps_grid = (self.scale_factor * coords.float()).to(torch.int)
        self.coords = coords

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample_paths = self.pairs[idx] 
        num_tupl = len(sample_paths)
        if num_tupl != self.n_tupl:
            raise ValueError(f"Expected {self.n_tupl} images, but got {num_tupl} images in the sample.")
        imgs = []
        masks = []
        img_sizes = []
        features_list = []
        sph_list = []
        for img_path in sample_paths:
            # Load image and resize
            img = Image.open(img_path).convert('RGB')
            img_sizes.append(img.size)
            img_resized = resize(img, self.img_size, resize=True, to_pil=True, edge=self.edge_pad)
            imgs.append(img_resized)
            # Load mask and resize
            mask_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{self.mask_suffix}.png")
            mask = Image.open(mask_path).convert('RGB')
            mask_resized = resize(mask, self.img_size, resize=True, to_pil=True, edge=self.edge_pad)
            masks.append(mask_resized)

            # Load features
            sph_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{self.sph_path_suffix}.pt")
            if self.filter_sph:
                img1_sph_load = torch.load(sph_path).detach() if os.path.exists(sph_path) else torch.tensor([])
                sph_list.append(img1_sph_load.squeeze())

            dino_feat_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{self.dino_path_suffix}.pt")
            if not self.only_dino:
                sd_feat_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{self.sd_path_suffix}.pt")
                img1_sd = torch.load(sd_feat_path)
            img1_dino = torch.load(dino_feat_path).detach()

            if self.only_dino:
                desc_gathered = img1_dino.reshape(1, -1, self.num_patches**2).permute(0, 2, 1)
            else:
                desc_gathered = torch.cat([
                    img1_sd['s3'].detach(),
                    F.interpolate(img1_sd['s4'].detach(), size=(self.num_patches, self.num_patches),mode='bilinear', align_corners=False),
                    F.interpolate(img1_sd['s5'].detach(), size=(self.num_patches, self.num_patches),mode='bilinear', align_corners=False),
                    img1_dino
                ], dim=1).reshape(1, -1, self.num_patches**2).permute(0, 2, 1).detach()
            features_list.append(normalize_features(desc_gathered, with_sd=not self.only_dino))
        features = torch.cat(features_list, dim=0)
        if self.filter_sph:
            sphs = torch.stack(sph_list, dim=0)
        
        foreground_inds = []
        for i_m, mask_img in enumerate(masks):
            if self.use_mask:
                mask_array_resized = resize(mask_img, self.num_patches, resize=True,
                                            to_pil=False, edge=self.edge_pad, sampling_filter='nearest')
                kp_on_obj = torch.tensor(np.argwhere(mask_array_resized > 0)[::3, :2], dtype=torch.float32)
                sal_idx = torch.nonzero((self.coords.unsqueeze(1) == kp_on_obj).all(-1).any(-1)).squeeze()
            else:
                img_array_resized = resize(imgs[i_m], self.num_patches, resize=True,
                                            to_pil=False, edge=self.edge_pad, sampling_filter='nearest')
                kp_not_padded = torch.tensor(np.argwhere(np.array(img_array_resized).sum(axis=-1) > 0)[:, :2], dtype=torch.float32)
                sal_idx = torch.nonzero((self.coords.unsqueeze(1) == kp_not_padded).all(-1).any(-1)).squeeze()
            
            foreground_inds.append(sal_idx)
        
        selected_pair_tuples = []
        if self.pseudo_gt_gen_mode == "nearest_neighbor":
            sel_idx_1, sel_idx_2 = get_cyclic_consistent_kps_for_pair(
                                    features, 0, 1, foreground_inds[0],foreground_inds[1],
                                    cyclic_consistent=self.cyclic_consistent,
                                    cyclic_consistency_mode=self.cyclic_consistency_mode,
                                    rej_th=self.rej_th_patch*self.scale_factor,
                                    kps_grid=self.kps_grid)
            selected_pair_tuples.append((0, 1, sel_idx_1, sel_idx_2, sample_paths[0], sample_paths[1]))
        elif self.pseudo_gt_gen_mode == 'bicyclic_chain':
            inds = list(range(num_tupl))
            sel_idx_12 = foreground_inds[0]
            all_inds = []
            for idx in inds[:-1]:
                i1,i2 = idx, idx+1
                sel_idx_11, sel_idx_12 = get_cyclic_consistent_kps_for_pair(
                                            features,i1,i2,sel_idx_12,foreground_inds[i2], 
                                            cyclic_consistent=True,
                                            cyclic_consistency_mode=self.cyclic_consistency_mode,
                                            rej_th=self.rej_th_patch*self.scale_factor,
                                            kps_grid=self.kps_grid,)
                source_kps = self.kps_grid[sel_idx_11].cpu(); target_kps = self.kps_grid[sel_idx_12].cpu()
                all_inds.append((sel_idx_11, sel_idx_12))
            final_inds_list = num_tupl*[None]
            sel_idx_11,sel_idx_12 = all_inds[-1]
            final_inds_list[-1] = sel_idx_12
            final_inds_list[-2] = sel_idx_11
            for i_i in range(-2,-num_tupl,-1):
                sel_idx_11,sel_idx_12 = all_inds[i_i]
                m_i = torch.isin(sel_idx_12, final_inds_list[i_i])
                final_inds_list[i_i-1] = sel_idx_11[m_i]

            sel_pairs = list(itertools.permutations(range(num_tupl), 2))
            random.shuffle(sel_pairs)
            for i, j in sel_pairs:
                selected_pair_tuples.append((i, j, final_inds_list[i], final_inds_list[j],sample_paths[i], sample_paths[j]))
        final_annotations = []
        for (i_idx, j_idx, sel_idx_1, sel_idx_2, src_im_path, tgt_im_path) in selected_pair_tuples:
            if self.filter_sph:
                m_sph_dist = filter_out_with_sph(sphs[i_idx, sel_idx_1],
                                                 sphs[j_idx, sel_idx_2],
                                                 thresh=self.filter_sph_threshold)
                sel_idx_1 = sel_idx_1[m_sph_dist]
                sel_idx_2 = sel_idx_2[m_sph_dist]
            source_kps = self.kps_grid[sel_idx_1]
            target_kps = self.kps_grid[sel_idx_2]
            if len(source_kps) == 0:
                continue
            ann = {
                'src_imname': os.path.basename(src_im_path),
                'trg_imname': os.path.basename(tgt_im_path),
                'src_imsize': img_sizes[i_idx],
                'tgt_imsize': img_sizes[j_idx],
                'num_kps': len(source_kps),
                'source_kps': source_kps.numpy()[:, [1, 0]].tolist(),
                'target_kps': target_kps.numpy()[:, [1, 0]].tolist(),
                'img_size_kps': self.img_size
            }
            final_annotations.append(ann)
        return final_annotations

def collate_fn(batch):
    return [ann for ann_list in batch if ann_list is not None for ann in ann_list]

def main(args):
    if args.batch_size % 2 != 0:
        raise ValueError("batch_size must be even")

    device = args.device if torch.cuda.is_available() else 'cpu'
    pseudo_gt_gen_mode = args.pseudo_gt_gen_mode

    os.makedirs(args.ann_dir, exist_ok=True)
    ann_dir = f"{args.ann_dir}/{args.dataset_version}/{args.split}"
    os.makedirs(ann_dir, exist_ok=True)

    df = aggregate_spair_data_in_df(args.spair_dir, split=args.split)

    chain_length = args.n_tupl

    # Process each category
    for cat in df.groupby(['category']).size().index:
        df_filt = df[df['category']==cat]
        if pseudo_gt_gen_mode == 'nearest_neighbor':
            subsample=args.subsample
            pairs = sample_spair_img_pairs(df_filt, subsample=subsample)
        elif pseudo_gt_gen_mode == 'bicyclic_chain':
            factorial_of_n = int(math.factorial(chain_length) / math.factorial(chain_length - 2))
            subsample = args.subsample // factorial_of_n + 1
            subsample = int(2 * subsample) # Ensure to have enough pairs
            pairs = sample_ordered_n_tupls(df_filt, n=chain_length, metric=args.metric_chaining, subsample=subsample)
        else:
            raise NotImplementedError(f"Pseudo GT generation method {pseudo_gt_gen_mode} does not exist.")
        pairs = random.sample(pairs, min(subsample, len(pairs)))
        print(f"Category {cat}... #images={len(df_filt)} #pairs={len(pairs)}")
        if len(pairs) == 0:
            print(f"No pairs found for {cat}")
            continue
        num_pairs = len(pairs)
        available_pairs = []
        for i_p in range(num_pairs):
            num_tupl = len(pairs[i_p])
            for i_d in range(num_tupl):
                img_path = pairs[i_p][i_d]
                img_exists = os.path.exists(img_path)
                mask_exists = os.path.exists(img_path.replace("JPEGImages", "features").replace(".jpg", f"{args.mask_suffix}.png"))
                dino_feat_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{args.dino_path_suffix}.pt")
                sd_feat_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{args.sd_path_suffix}.pt")
                sph_path = img_path.replace("JPEGImages", "features").replace(".jpg", f"{args.sph_path_suffix}.pt")
                dino_exists, sd_exists, sph_exists = os.path.exists(dino_feat_path), os.path.exists(sd_feat_path), os.path.exists(sph_path)
                feat_exists = dino_exists
                if not args.only_dino:
                    feat_exists = feat_exists and sd_exists
                if args.filter_sph:
                    feat_exists = feat_exists and sph_exists
                if not feat_exists:
                    print(f"img_exists: {img_exists}, dino_exists: {dino_exists}, sd_exists: {sd_exists}, sph_exists: {sph_exists}, mask_exists: {mask_exists} for {img_path}")
                valid = img_exists and feat_exists and mask_exists
            if valid:
                available_pairs.append(pairs[i_p])

        dataset = PairsDataset(
            available_pairs,
            img_size=args.img_size,
            edge_pad=args.edge_pad,
            num_patches=args.num_patches,
            device=device,
            pseudo_gt_gen_mode=pseudo_gt_gen_mode,
            only_dino=args.only_dino,
            cyclic_consistent=args.cyclic_consistent,
            cyclic_consistency_mode=args.cyclic_consistency_mode,
            rej_th_patch=args.rej_th_patch,
            use_mask=not args.no_mask,
            n_tupl=args.n_tupl,
            filter_sph=args.filter_sph,
            filter_sph_threshold=args.filter_sph_threshold,
            dino_path_suffix=args.dino_path_suffix,
            sd_path_suffix=args.sd_path_suffix,
            sph_path_suffix=args.sph_path_suffix,
            mask_suffix=args.mask_suffix,
        )
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=4, collate_fn=collate_fn)
        
        sample_count = 0
        anns_cat = []
        for ann_batch in tqdm(dataloader, desc=f"Processing batches for {cat}"):
            # Each ann_batch is a list of annotation dictionaries (one per n-tuple)
            for ann in ann_batch:
                anns_cat.append(ann)
                sample_count += 1
                if sample_count >= args.subsample:
                    break
            if sample_count >= args.subsample:
                    break
        anns_cat_file = f"{ann_dir}/{cat}_anns.pt"
        torch.save(anns_cat, anns_cat_file)
        print(f"Saved {len(anns_cat)} annotations for category {cat} to {anns_cat_file}.")


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Pseudo GT Image Pairs Generation")
    # Experiment details
    parser.add_argument('--dataset_version', type=str, default="vXX", help='Dataset version')
    parser.add_argument('--split', type=str, default="trn", choices=['trn', 'val', 'test'], help='Split [trn, val, test]')
    parser.add_argument('--subsample', type=int, default=1000, help='Number of considered pairs')
    parser.add_argument('--ann_dir', type=str, default="data/pair_annotations/spair", help='Annotation directory')

    # Loading configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--edge_pad', action='store_true', default=False, help='Whether to use edge padding')
    parser.add_argument('--img_size', type=int, default=840, help='Image size for resizing')
    parser.add_argument('--num_patches', type=int, default=60, help='Number of patches for feature extraction')
    parser.add_argument('--spair_dir', type=str, default="data/SPair-71k/", help='SPair directory')
    parser.add_argument('--dino_path_suffix', type=str, default='_dino', help='DINO feature path suffix')
    parser.add_argument('--sd_path_suffix', type=str, default='_sd', help='SD feature path suffix')
    parser.add_argument('--sph_path_suffix', type=str, default='_sph', help='sph path suffix')
    parser.add_argument('--mask_suffix', type=str, default='_mask', help='mask path suffix')
    
    # Pseudo GT generation configuration
    parser.add_argument('--pseudo_gt_gen_mode', type=str, default="bicyclic_chain", choices=['bicyclic_chain', 'nearest_neighbor'], help='Pseudo-label generation method.')
    parser.add_argument('--only_dino', action='store_true', default=False, help='Whether to only use DINO for the pseudo-labels.')
    parser.add_argument('--cyclic_consistency_mode', type=str, default="eucl_dist", help='Mode for cyclic consistency.')
    parser.add_argument('--n_tupl', type=int, default=4, help='Number of samples in n-tuples.')
    parser.add_argument('--metric_chaining', type=str, default='azimuth_bin', help='Quantity used for ranking along the chain.')
    parser.add_argument('--filter_sph', action='store_true', default=False, help='Filter with spherical mapper.')
    parser.add_argument('--filter_sph_threshold', type=float, default=0.15, help='Filter with spherical mapper threshold.')
    parser.add_argument('--rej_th_patch', type=int, default=2, help='Relaxed cyclic consistency threshold.')
    parser.add_argument('--no_mask', action='store_true', default=False, help='Do not use mask for pseudo labels.')
    parser.add_argument('--cyclic_consistent', action='store_true', default=False, help='Whether to do it cyclic consistent.')
    
    args = parser.parse_args()
    print(args)
    main(args)
