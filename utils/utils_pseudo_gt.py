import numpy as np
import torch
from PIL import Image
import random
import itertools
import os
import json
import pandas as pd
from tqdm import tqdm
from glob import glob


def sample_spair_img_pairs(df_filt, d_azimuth=None, subsample=None):
   
    if d_azimuth is not None:
        azimuths = df_filt['azimuth_bin'].values
        filt_dist = np.abs(azimuths[:, None] - azimuths[None, :])
        filt_dist = np.minimum(filt_dist, 8 - filt_dist)
        np.fill_diagonal(filt_dist, 1e6)
    else:
        filt_dist = np.zeros((df_filt.shape[0],df_filt.shape[1]))
        d_azimuth = 1

    m_low_dist = filt_dist < d_azimuth
    rows,cols = np.where(m_low_dist)
    pairs = list(zip(df_filt.iloc[rows]['img_path'], df_filt.iloc[cols]['img_path']))
    if subsample is not None:
        if subsample < len(pairs):
            rand_inds = random.sample(range(len(pairs)), int(subsample))
            pairs = [pairs[ri] for ri in rand_inds]
    return pairs


def filter_out_with_sph(feati,featj, thresh=0.25):
    cosine_sim = torch.nn.functional.cosine_similarity(feati, featj, dim=-1)
    distances_cosine = torch.acos(cosine_sim.clamp(-1, 1))
    m_sph_dist = distances_cosine <= (torch.pi * thresh)
    return m_sph_dist


def sample_ordered_n_tupls(df_filt,n=4, metric='random',subsample=0, num_bins=8):
    df_filt = df_filt.sort_values('img_path')
    img_paths = df_filt['img_path'].values
    if metric=='random':
        indices = [tuple(random.sample(range(df_filt.shape[0]), n)) for _ in range(subsample)]
        all_n_tupls_inds = [s for s in indices if len(set(s))==n]
    elif metric=='azimuth_bin':
        azimuths = df_filt[metric].values
        if len(np.unique(azimuths))<n: # Sample randomly because not sufficiently large azimuth variation.
            indices = [tuple(random.sample(range(df_filt.shape[0]), n)) for _ in range(subsample)]
            all_n_tupls_inds = [s for s in indices if len(set(s))==n]
        else:
            all_n_tupls_inds = []
            for sign_change in [1,-1]:
                es = []
                filt_dist = (azimuths[None, :] - azimuths[:, None]) % (sign_change*num_bins)
                for j in range(len(filt_dist)):
                    row_filt = filt_dist[j]
                    es.append([])
                    es[j].append([j])
                    for i in range(1, n):
                        e = np.where(row_filt == (sign_change*i))[0]
                        es[j].append(e)
                for j in range(len(filt_dist)):
                    all_n_tupls_inds += list(itertools.product(*es[j]))
    else:
        raise NotImplementedError('Chaining metric not implemented.')
    
    if subsample > 0:
        all_n_tupls_inds = random.sample(all_n_tupls_inds, min(subsample,len(all_n_tupls_inds)))
    all_n_tupls = [[img_paths[i] for i in inds] for inds in all_n_tupls_inds]

    return all_n_tupls


def get_cyclic_consistent_kps_for_pair(feat_desc,idx1,idx2,sal_idx_1,sal_idx_2, cyclic_consistent=True, cyclic_consistency_mode='strict',
                                        rej_th=2,kps_grid=None):

    # 1_sal -> 2_sal
    nn_1_in_2 = nn_j_from_i(feat_desc[idx1,sal_idx_1], feat_desc[idx2,sal_idx_2])
    sal_idx_2_from_1 = sal_idx_2[nn_1_in_2]

    if cyclic_consistent:
        # (1_sal->2_sal) -> 1_sal
        nn_2_in_1 = nn_j_from_i(feat_desc[idx2,sal_idx_2_from_1], feat_desc[idx1,sal_idx_1])
        sal_idx_1_from_2 = sal_idx_1[nn_2_in_1]
        # filter out non-cyclic consistent correspondences
        if cyclic_consistency_mode == 'strict':
            cyclic_consisent = sal_idx_1 == sal_idx_1_from_2
        elif cyclic_consistency_mode == 'eucl_dist':
            if kps_grid is None:
                raise ValueError('kps_grid is None')
            sal_kps_1 = kps_grid[sal_idx_1.long()].cpu()
            sal_kps_12 = kps_grid[sal_idx_1_from_2.long()].cpu()
            distances = torch.norm(sal_kps_1.float() - sal_kps_12.float(), dim=1)
            cyclic_consisent = distances <= rej_th
        else:
            raise NotImplementedError(f'cyclic_consistenty_mode {cyclic_consistency_mode} not implemented')

        sel_idx_1 = sal_idx_1[cyclic_consisent]
        sel_idx_2 = sal_idx_2_from_1[cyclic_consisent]
    else: # baseline
        sel_idx_1 = sal_idx_1
        sel_idx_2 = sal_idx_2_from_1

    return sel_idx_1, sel_idx_2


def nn_j_from_i(feati, featj):
    sim_ij = torch.matmul(feati, featj.permute(1,0))
    nn_ij = torch.argmax(sim_ij, dim=-1).cpu()
    return nn_ij


def normalize_features(feats, epsilon=1e-10, with_sd=True):
    if with_sd:
        feat_sd = feats[..., :640+1280+1280]
        feat_dino = feats[..., 640+1280+1280:]
        norms_sd = torch.linalg.norm(feat_sd, dim=-1)[:, :, None]
        norm_feats_sd = feat_sd / (norms_sd + epsilon)
        norms_dino = torch.linalg.norm(feat_dino, dim=-1)[:, :, None]
        norm_feats_dino = feat_dino / (norms_dino + epsilon)
        feats = torch.cat([norm_feats_sd, norm_feats_dino], dim=-1)
    # (b, w*h, c)
    norms = torch.linalg.norm(feats, dim=-1)[:, :, None]
    norm_feats = feats / (norms + epsilon)
    return norm_feats


def resize(img, target_res=224, resize=True, to_pil=True, edge=False, sampling_filter='lanczos', padding=True):
    filt = Image.Resampling.LANCZOS if sampling_filter == 'lanczos' else Image.Resampling.NEAREST
    if not padding and resize:
        return img.resize((target_res, target_res), filt)
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), filt)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), filt)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), filt)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), filt)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def aggregate_spair_data_in_df(data_dir, split):
    categories = sorted(os.listdir(os.path.join(data_dir, 'JPEGImages')))

    df_collect = []
    for cat in tqdm(categories, desc="Processing categories"):
        category_annos = glob(os.path.join(data_dir, 'ImageAnnotation', cat, '*.json'))
        split_pairs = glob(os.path.join(data_dir, 'PairAnnotation', split, '*.json'))
        split_pairs_cat = [t for t in split_pairs if cat in t]

        split_imgs = [pair.split('/')[-1].split(':')[0].split('-')[1:] for pair in split_pairs_cat]
        split_imgs_flat = [img for sublist in split_imgs for img in sublist]

        for cat_ann in category_annos:
            if os.path.splitext(os.path.basename(cat_ann))[0] not in split_imgs_flat:
                continue
            with open(cat_ann) as f:
                d = json.load(f)
            img_path_i = os.path.join(data_dir, 'JPEGImages', cat, d['filename'])
            df_collect.append([cat,d['difficult'],d['occluded'],d['truncated'],d['azimuth_id'],d['pose'],d['filename'],img_path_i])

    df_collect = pd.DataFrame(df_collect, columns=[
            'category', 'difficult', 'occluded', 'truncated', 'azimuth_bin', 'pose', 'filename', 'img_path'
        ])
    return df_collect