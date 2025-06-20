import os
import torch
import pickle
import wandb
import argparse
import configargparse
from PIL import Image
from tqdm import tqdm
from loguru import logger
import time
from itertools import chain
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from preprocess_map import set_seed
from model_utils.projection_network import AggregationNetwork, DummyAggregationNetwork
from model_utils.corr_map_model import Correlation2Displacement
import utils.utils_losses as utils_losses
import utils.utils_visualization as utils_visualization
from utils.logger import get_logger, log_geo_stats, update_stats, update_geo_stats, log_weighted_pcks, load_config
from utils.utils_geoware import AP10K_GEO_AWARE, AP10K_FLIP, SPAIR_GEO_AWARE, SPAIR_FLIP, SPAIR_FLIP_TRN, permute_indices, renumber_indices, flip_keypoints, renumber_used_points, optimized_kps_1_to_2
from utils.utils_correspondence import kpts_to_patch_idx, load_img_and_kps, calculate_keypoint_transformation, get_distance, get_distance_mutual_nn, normalize_feats, load_features_and_masks
from utils.utils_dataset import load_eval_data, load_and_prepare_data, get_dataset_info

device  = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=False, flip2=False, img1=None, img2=None, device='cuda'):
    img_path_1 = files[pair_idx * 2]
    img_path_2 = files[pair_idx * 2 + 1]
    # save the imgs for cases if the feature doesn't exist
    img1_desc, mask1 = load_features_and_masks(aggre_net, img_path_1, flip, args.ENSEMBLE, num_patches, device,img_dir=args.IMG_DIR,feature_dir=args.FEATURE_DIR, only_dino=args.ONLY_DINO, sd_path_suffix=args.SD_PATH_SUFFIX, dino_path_suffix=args.DINO_PATH_SUFFIX)
    img2_desc, mask2 = load_features_and_masks(aggre_net, img_path_2, flip2, args.ENSEMBLE, num_patches, device,img_dir=args.IMG_DIR,feature_dir=args.FEATURE_DIR, only_dino=args.ONLY_DINO, sd_path_suffix=args.SD_PATH_SUFFIX, dino_path_suffix=args.DINO_PATH_SUFFIX)
    
    # normalize the desc
    img1_desc = normalize_feats(img1_desc[0], sd_and_dino=args.DUMMY_NET and not args.ONLY_DINO)
    img2_desc = normalize_feats(img2_desc[0], sd_and_dino=args.DUMMY_NET and not args.ONLY_DINO)
    return img1_desc, img2_desc, mask1, mask2

def compute_pck(args, save_path, aggre_net, files, kps, category=None, used_points=None, thresholds=None):
    out_results = []
    num_patches = args.NUM_PATCHES
    current_save_results = 0
    gt_correspondences, pred_correspondences, img_acc_001, img_acc_005, img_acc_01, len_kpts = ([] for _ in range(6))
    if thresholds is not None:
        thresholds = torch.tensor(thresholds).to(device)
        bbox_size=[]
    N = len(files) // 2
    
    vis_inds = torch.randperm(N)[:args.TOTAL_SAVE_RESULT] if args.TOTAL_SAVE_RESULT > 0 else []

    if args.COMPUTE_GEOAWARE_METRICS:   # get the geo-aware idx list
        geo_aware_count = geo_aware_total_count = 0
        geo_idx_all, influ_list_geo_filtered = [], []
        if args.EVAL_DATASET == 'ap10k':
            influ_list_geo = AP10K_GEO_AWARE
        elif args.EVAL_DATASET == 'in3d':
            influ_list_geo = []
        else:
            influ_list_geo = SPAIR_GEO_AWARE[category] if category in SPAIR_GEO_AWARE else None
        for item in influ_list_geo:
            item = [item] if isinstance(item, int) else item
            temp_list = [idx for idx in item if idx in used_points]
            if len(temp_list) >= 1:
                influ_list_geo_filtered.append(temp_list)
        raw_geo_aware = renumber_indices(influ_list_geo_filtered, counter=[0])
    
    if args.ADAPT_FLIP: # get the permute list for flipping
        FLIP_ANNO = AP10K_FLIP if args.EVAL_DATASET == 'ap10k' else SPAIR_FLIP[category]
        if sum(len(i) if isinstance(i, list) else 1 for i in FLIP_ANNO) == kps[0].shape[0]:
            permute_list = FLIP_ANNO
        else:
            influ_list_filtered = []
            influ_list = FLIP_ANNO
            for item in influ_list:
                item = [item] if isinstance(item, int) else item
                temp_list = [idx for idx in item if idx in used_points]
                if len(temp_list) >= 1:
                    influ_list_filtered.append(temp_list)
            permute_list = renumber_indices(influ_list_filtered, counter=[0])

    for pair_idx in range(N):
        # Load images and keypoints
        img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, img_size=args.ANNO_SIZE, edge=False)
        # Get mutual visibility
        vis = img1_kps[:, 2] * img2_kps[:, 2] > 0
        vis2 = img2_kps[:, 2]
        # Get patch descriptors
        with torch.no_grad():
            img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2, device=device)
        # Get patch index for the keypoints
        img1_patch_idx = kpts_to_patch_idx(args, img1_kps, num_patches)
        # Get similarity matrix
        kps_1_to_2 = calculate_keypoint_transformation(args, img1_desc, img2_desc, img1_patch_idx, num_patches)

        if args.ADAPT_FLIP:
            img1_flip = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img1_desc_flip, _, mask1_flip, _ = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2, device=device)
            img1_kps_flip = flip_keypoints(img1_kps, args.ANNO_SIZE, permute_indices(permute_list, vis))
            img1_patch_idx_flip = kpts_to_patch_idx(args, img1_kps_flip, num_patches)
            kps_1_to_2_flip = calculate_keypoint_transformation(args, img1_desc_flip, img2_desc, img1_patch_idx_flip, num_patches)
            
            # get the distance for the flip and original img
            if args.MUTUAL_NN:
                original_dist = get_distance_mutual_nn(img1_desc, img2_desc)
                flip_dist = get_distance_mutual_nn(img1_desc_flip, img2_desc)
            else:
                original_dist = get_distance(img1_desc, img2_desc, mask1, mask2)
                flip_dist = get_distance(img1_desc_flip, img2_desc, mask1_flip, mask2)

            kps_1_to_2 = optimized_kps_1_to_2(args, kps_1_to_2, kps_1_to_2_flip, img1_kps, img2_kps, flip_dist, original_dist, vis, permute_list)

        # collect the result for more complicated eval
        single_result = {
            "src_fn": files[2*pair_idx],  # must
            "trg_fn": files[2*pair_idx+1],  # must
            "category": category,
            "used_points": used_points.cpu().numpy(),
            "src_kpts": renumber_used_points(img1_kps, used_points).cpu().numpy(),
            "trg_kpts": renumber_used_points(img2_kps, used_points).cpu().numpy(),
            "src_kpts_pred": renumber_used_points(kps_1_to_2.cpu(), used_points).cpu().detach().numpy(),  # must
            "threshold": thresholds[pair_idx].item() if thresholds is not None else 0,
            "resize_resolution": args.ANNO_SIZE,  # must
        }
        out_results.append(single_result)

        gt_kps = img2_kps[vis][:, [1,0]]
        prd_kps = kps_1_to_2[vis][:, [1,0]]
        gt_correspondences.append(gt_kps)
        pred_correspondences.append(prd_kps)
        len_kpts.append(vis.sum().item())

        # compute per image acc
        if not args.KPT_RESULT: # per img result
            single_gt_correspondences = img2_kps[vis][:, [1,0]]
            single_pred_correspondences = kps_1_to_2[vis][:, [1,0]]
            alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
            correct = torch.zeros(3)
            err = (single_gt_correspondences - single_pred_correspondences.cpu()).norm(dim=-1)
            err = err.unsqueeze(0).repeat(3, 1)
            if thresholds is not None:
                single_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
                correct += (err < alpha.unsqueeze(-1) * single_bbox_size.unsqueeze(0)).float().mean(dim=-1)
            else:
                correct += (err < alpha.unsqueeze(-1) * args.ANNO_SIZE).float().mean(dim=-1)
            img_acc_01.append(correct[0].item())
            img_acc_005.append(correct[1].item())
            img_acc_001.append(correct[2].item())

        if thresholds is not None:
            pckthres = thresholds[pair_idx].repeat(vis.sum())
            bbox_size.append(pckthres)

        if args.COMPUTE_GEOAWARE_METRICS:
            geo_aware_list, geo_aware_full_list = ([] for _ in range(2))
            for item in raw_geo_aware:
                # convert to list
                item = [item] if isinstance(item, int) else item
                # check if all items are visible
                temp_list = [idx for idx in item if vis[idx]]
                temp_list2 = [idx for idx in item if vis2[idx]]
                # if more than 2 items are visible, add to geo_aware_list
                if len(temp_list2) >= 2 and len(temp_list) >= 1:
                    for temp_idx in temp_list:
                        geo_aware_list.append([temp_idx])
                    geo_aware_full_list.append(temp_list)
            
            geo_aware_idx = [item for sublist in geo_aware_list for item in sublist]
            geo_idx_mask = torch.zeros(len(vis)).bool()
            geo_idx_mask[geo_aware_idx] = True
            geo_idx_mask = geo_idx_mask[vis]
            geo_idx_all.append(geo_idx_mask.clone().detach())
            
            # count the number of geo-aware pairs
            if len(geo_aware_full_list) > 0: 
                geo_aware_total_count += len(geo_aware_idx)     # per keypoint
                geo_aware_count += 1                            # per img
            
        if pair_idx in vis_inds:
            if args.ADAPT_FLIP and (flip_dist < original_dist): # save the flip result
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category, 
                       img1_kps_flip, img1_flip, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            else:
                utils_visualization.save_visualization(thresholds, pair_idx, vis, save_path, category, 
                       img1_kps, img1, img2, kps_1_to_2, img2_kps, args.ANNO_SIZE, args.ADAPT_FLIP)
            current_save_results += 1

        # pbar.update(1)
    if not args.KPT_RESULT:
        img_correct = torch.tensor([img_acc_01, img_acc_005, img_acc_001])
        img_correct = img_correct.mean(dim=-1).tolist()
        img_correct.append(N)
    else:
        img_correct = None
    gt_correspondences = torch.cat(gt_correspondences, dim=0).cpu()
    pred_correspondences = torch.cat(pred_correspondences, dim=0).cpu()
    alpha = torch.tensor([0.1, 0.05, 0.01]) if args.EVAL_DATASET != 'pascal' else torch.tensor([0.1, 0.05, 0.15])
    correct = torch.zeros(len(alpha))
    err = (pred_correspondences - gt_correspondences).norm(dim=-1)
    err = err.unsqueeze(0).repeat(len(alpha), 1)
    if thresholds is not None:
        bbox_size = torch.cat(bbox_size, dim=0).cpu()
        threshold = alpha.unsqueeze(-1) * bbox_size.unsqueeze(0)
        correct_all = err < threshold
    else:
        threshold = alpha * args.ANNO_SIZE
        correct_all = err < threshold.unsqueeze(-1)

    correct = correct_all.sum(dim=-1) / len(gt_correspondences)
    correct = correct.tolist()
    correct.append(len(gt_correspondences))
    alpha2pck = zip(alpha.tolist(), correct[:3]) if args.KPT_RESULT else zip(alpha.tolist(), img_correct[:3])
    logger.info(f'{category}...'+' | '.join([f'PCK-Transfer@{alpha:.2f}: {pck_alpha * 100:.2f}%'
        for alpha, pck_alpha in alpha2pck]))
    
    geo_score = []
    if args.COMPUTE_GEOAWARE_METRICS:
        geo_idx_all = torch.cat(geo_idx_all, dim=0).cpu()
        correct_geo = correct_all[:,geo_idx_all].sum(dim=-1) / geo_idx_all.sum().item()
        correct_geo = correct_geo.tolist()
        geo_score.append(geo_aware_count / N)
        geo_score.append(geo_aware_total_count / len(gt_correspondences))
        geo_score.extend(correct_geo)
        geo_score.append(geo_idx_all.sum().item())
        alpha2pck_geo = zip(alpha.tolist(), correct_geo[:3])
        logger.info(' | '.join([f'PCK-Transfer_geo-aware@{alpha:.2f}: {pck_alpha * 100:.2f}%'
                        for alpha, pck_alpha in alpha2pck_geo]))
        logger.info(f'Geo-aware occurance count: {geo_aware_count}, with ratio {geo_aware_count / N * 100:.2f}%; total count ratio {geo_aware_total_count / len(gt_correspondences) * 100:.2f}%')

    return correct, geo_score, out_results, img_correct

def train(args, aggre_net, corr_map_net, optimizer, scheduler, logger, save_path):
    # gather training data
    files, kps, _, _, all_thresholds,flips = load_and_prepare_data(args)
    # train
    num_patches = args.NUM_PATCHES
    N = len(files) // 2
    pbar = tqdm(total=N)
    max_pck_010 = max_pck_005 = max_pck_001 = max_iter = loss_count = count = 0
    best_val_loss = 1e10
    if not args.NOT_VAL: best_val_loss = 10
    i_step = 0
    stop_training = False
    for epoch in range(args.EPOCH):
        if stop_training: break
        pbar.reset()
        for j in range(0, N, args.BZ):
            if j + epoch * N >= args.MAX_STEPS:
                logger.info(f'Stopping after {i_step} steps and {j + epoch * N} samples.')
                stop_training = True
                break
            optimizer.zero_grad()
            batch_loss = 0  # collect the loss for each batch
            for pair_idx in range(j, min(j+args.BZ, N)):
                # Load images and keypoints
                img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, edge=False)
                img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, edge=False)
                # Get patch descriptors/feature maps
                img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, img1=img1, img2=img2, device=device)
                if args.ADAPT_FLIP > 0 or args.AUGMENT_SELF_FLIP > 0 or args.AUGMENT_DOUBLE_FLIP > 0:  # augment with flip
                    img1_desc_flip, img2_desc_flip, _, _ = get_patch_descriptors(args, aggre_net, num_patches, files, pair_idx, flip=True, flip2=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2.transpose(Image.FLIP_LEFT_RIGHT), device=device)
                    raw_permute_list = AP10K_FLIP if args.TRAIN_DATASET == 'ap10k' else SPAIR_FLIP_TRN[files[pair_idx * 2].split('/')[-2]]
                else:
                    img1_desc_flip = img2_desc_flip = raw_permute_list = None
                
                # Get the threshold for each patch
                scale_factor = num_patches / args.ANNO_SIZE
                if args.BBOX_THRE:
                    img1_threshold = all_thresholds[2*pair_idx] * scale_factor
                    img2_threshold = all_thresholds[2*pair_idx+1] * scale_factor
                else: # image threshold
                    img1_threshold = img2_threshold = args.ANNO_SIZE

                # Compute loss
                loss = utils_losses.calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc, img1_threshold, img2_threshold, mask1, mask2, 
                                                   num_patches, device, raw_permute_list, img1_desc_flip, img2_desc_flip, corr_map_net, flips[2*pair_idx+1])

                # Accumulate loss over iterations
                loss_count += loss.item()
                count += 1
                batch_loss += loss
                pbar.update(1)

                cur_idx = pair_idx + epoch * N

                with torch.no_grad():
                    # Log loss periodically or at the end of the dataset
                    if (cur_idx % 100 == 0 and pair_idx > -1) or pair_idx == N-1: # Log every 100 iterations and at the end of the dataset
                        logger.info(f'Step {pair_idx + epoch * N} | Loss: {loss_count / count:.4f}')
                        wandb_dict = {'loss': loss_count / count}
                        loss_count = count = 0 # reset loss count
                        if not args.NOT_WANDB: wandb.log(wandb_dict, step=pair_idx + epoch * N)
                    # Evaluate model periodically, at the end of the dataset, or under specific conditions
                    if (cur_idx % args.EVAL_EPOCH == 0 and pair_idx >0): # or pair_idx == N-1:  # Evaluate every args.EVAL_EPOCH iterations and at the end of the dataset
                        if not args.NOT_VAL:
                            avg_val_loss = perform_eval(args, aggre_net,corr_map_net)
                            logger.info(f'Current val loss: {avg_val_loss:.4f}')
                        else:
                            avg_val_loss = best_val_loss
                        pck_010, pck_005, pck_001, total_result = eval(args, aggre_net, save_path)  # Perform evaluation
                        logger.info(f'Current PCK0.10: {pck_010 * 100:.2f}% at step {pair_idx + epoch * N}')
                        wandb_dict = {'pck_010': pck_010, 'pck_005': pck_005, 'pck_001': pck_001, 'val_loss': avg_val_loss}
                        # Update best model based on PCK scores and dataset type
                        # if (pck_010 > max_pck_010 and args.EVAL_DATASET != 'pascal') or (pck_005 > max_pck_005 and args.EVAL_DATASET == 'pascal'): # different criteria for PASCAL_EVAL
                        #     max_pck_010, max_pck_005, max_pck_001 = pck_010, pck_005, pck_001
                        # Update best model based on val loss
                        if best_val_loss > avg_val_loss:
                            best_val_loss = avg_val_loss
                            max_pck_010, max_pck_005, max_pck_001 = pck_010, pck_005, pck_001
                            max_iter = pair_idx + epoch * N
                            torch.save(aggre_net.state_dict(), f'{save_path}/best.pth') # Save the best model
                        else:
                            torch.save(aggre_net.state_dict(), f'{save_path}/last.pth') # Save the last model if it's not the best
                        # Log the best PCK scores
                        logger.info(f'Best val loss: {best_val_loss:.4f} at step {pair_idx + epoch * N}')
                        logger.info(f'Best PCK0.10: {max_pck_010 * 100:.2f}% at step {pair_idx + epoch * N}, with PCK0.05: {max_pck_005 * 100:.2f}%, PCK0.01: {max_pck_001 * 100:.2f}%')
                        if not args.NOT_WANDB: wandb.log(wandb_dict, step=pair_idx + epoch * N)

            batch_loss /= args.BZ
            batch_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            i_step += 1

def perform_eval(args, aggre_net,corr_map_net):
    print(f"Performing validation...")
    t0 = time.time()
    aggre_net.eval()

    files, kps, _, _, all_thresholds,flips = load_and_prepare_data(args, val=True)
    
    loss_count = 0.
    count = 0
    with torch.no_grad():
        for pair_idx in range(0, len(files) // 2):
            # Load images and keypoints
            img1, img1_kps = load_img_and_kps(idx=2*pair_idx, files=files, kps=kps, edge=False)
            img2, img2_kps = load_img_and_kps(idx=2*pair_idx+1, files=files, kps=kps, edge=False)
            # Get patch descriptors/feature maps
            img1_desc, img2_desc, mask1, mask2 = get_patch_descriptors(args, aggre_net, args.NUM_PATCHES, files, pair_idx, img1=img1, img2=img2, device=device)
            if args.ADAPT_FLIP > 0 or args.AUGMENT_SELF_FLIP > 0 or args.AUGMENT_DOUBLE_FLIP > 0:  # augment with flip
                img1_desc_flip, img2_desc_flip, _, _ = get_patch_descriptors(args, aggre_net, args.NUM_PATCHES, files, pair_idx, flip=True, flip2=True, img1=img1.transpose(Image.FLIP_LEFT_RIGHT), img2=img2.transpose(Image.FLIP_LEFT_RIGHT), device=device)
                raw_permute_list = AP10K_FLIP if args.TRAIN_DATASET == 'ap10k' else SPAIR_FLIP_TRN[files[pair_idx * 2].split('/')[-2]]
            else:
                img1_desc_flip = img2_desc_flip = raw_permute_list = None

            # Get the threshold for each patch
            scale_factor = args.NUM_PATCHES / args.ANNO_SIZE
            if args.BBOX_THRE:
                img1_threshold = all_thresholds[2*pair_idx] * scale_factor
                img2_threshold = all_thresholds[2*pair_idx+1] * scale_factor
            else: # image threshold
                img1_threshold = img2_threshold = args.ANNO_SIZE

            # Compute loss
            loss = utils_losses.calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc, img1_threshold, img2_threshold, mask1, mask2, 
                                                   args.NUM_PATCHES, device, raw_permute_list, img1_desc_flip, img2_desc_flip, corr_map_net, flips[2*pair_idx+1])

            # Accumulate loss over iterations
            loss_count += loss.item()
            count += 1

    avg_loss = loss_count / count

    aggre_net.train()
    t1 = time.time()
    print(f"Finished validation after {t1-t0:.2f}s.")
    return avg_loss

def eval(args, aggre_net, save_path, split='val'):
    aggre_net.eval()  # Set the network to evaluation mode
    # Configure data directory and categories based on the dataset type
    data_dir, categories, split = get_dataset_info(args, split)

    # Initialize lists for results and statistics
    total_out_results, pcks, pcks_05, pcks_01, weights, kpt_weights = ([] for _ in range(6))
    if args.COMPUTE_GEOAWARE_METRICS: geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo = ([] for _ in range(6))

    # Process each category
    for cat in categories:
        # Load data based on the dataset
        files, kps, thresholds, used_points = load_eval_data(args, data_dir, cat, split)
        # Compute PCK with or without bbox threshold
        compute_args = (save_path, aggre_net, files, kps, cat, used_points)
        pck, correct_geo, out_results, img_correct = compute_pck(args, *compute_args, thresholds=thresholds) if args.BBOX_THRE else compute_pck(args, *compute_args)
        total_out_results.extend(out_results)
        update_stats(args, pcks, pcks_05, pcks_01, weights, kpt_weights, pck, img_correct)
        if args.COMPUTE_GEOAWARE_METRICS and sum(geo_aware_count) > 0: 
            update_geo_stats(geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, correct_geo)

    # Calculate and log weighted PCKs
    pck_010, pck_005, pck_001 = log_weighted_pcks(args, logger, pcks, pcks_05, pcks_01, weights)
    if args.COMPUTE_GEOAWARE_METRICS and sum(geo_aware_count) > 0: 
        log_geo_stats(args, geo_aware, geo_aware_count, pcks_geo, pcks_geo_05, pcks_geo_01, weights_geo, kpt_weights, total_out_results)

    aggre_net.train()  # Set the network back to training mode
    return pck_010, pck_005, pck_001, total_out_results

def main(args):
    set_seed(args.SEED)
    args.NUM_PATCHES = 60
    args.BBOX_THRE = not (args.IMG_THRESHOLD or args.EVAL_DATASET == 'pascal')
    args.AUGMENT_FLIP, args.AUGMENT_DOUBLE_FLIP, args.AUGMENT_SELF_FLIP = (1.0, 1.0, 0.25) if args.PAIR_AUGMENT else (0, 0, 0) # set different weight for different augmentation
    if args.SAMPLE == 0: args.SAMPLE = None # use all the data
    feature_dims = [640,1280,1280,768] # dimensions for three layers of SD and one layer of DINOv2 features
    if args.ONLY_DINO: 
        feature_dims = [768,]

    # Determine the evaluation type and project name based on args
    available_exps = sorted(os.listdir('./results/exps'))
    used_exp_ids = [int(exp.split('_')[1]) for exp in available_exps if exp.startswith('exp_')]

    if not args.DO_EVAL:
        if args.EXP_ID in used_exp_ids:
            args.EXP_ID = max(used_exp_ids) + 1
            print(f'Experiment ID already exists. Setting EXP_ID to {args.EXP_ID}.')

    exp_name = f'exp_{args.EXP_ID:04d}_{args.NOTE}'

    if args.DO_EVAL and (not args.DUMMY_NET):
        if args.EXP_ID not in used_exp_ids:
            print(f'Experiment ID {args.EXP_ID:04d} does not exist.')
        if exp_name not in available_exps: 
            print(f'Experiment {exp_name} does not exist.')
        
    print(f'Experiment name: {exp_name}')
    
    if args.DO_EVAL:
        os.makedirs(f'./results/{args.EVAL_DATASET}', exist_ok=True)
        save_path = f'./results/{args.EVAL_DATASET}/{exp_name}'
    else:
        save_path = f'./results/exps/{exp_name}'
    results_name = f'results__sample_{args.TEST_SAMPLE}_kpt_{args.KPT_RESULT}_softeval_{args.SOFT_EVAL}_softevalwindow_{args.SOFT_EVAL_WINDOW}_dummy_{args.DUMMY_NET}_flip_{args.ADAPT_FLIP}'

    log_file = save_path+f'/{results_name}.log'
    
    i = 1; log_file_2 = log_file
    while os.path.exists(log_file_2):
        results_name_2 = f"{results_name}_{i:02d}" ; i += 1
        print(f'Log file {log_file_2} exists. Trying with {results_name_2}')
        log_file_2 = save_path+f'/{results_name_2}.log'
    if i > 1:
        results_name = results_name_2

    print(f'Saving results to {save_path} with {results_name}')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not args.NOT_WANDB:
        wandb.init(project=args.EVAL_DATASET, name=f'{exp_name}', config=args)
    
    logger = get_logger(save_path+f'/{results_name}.log')
    logger.info(args)
    
    logger.info(f'Saving results to {save_path} with {results_name}')

    if args.DUMMY_NET:
        aggre_net = DummyAggregationNetwork()
    else:
        print(f'Using feature dims: {feature_dims}, projection dim: {args.PROJ_DIM}, feat map dropout: {args.FEAT_MAP_DROPOUT}, device: {device}')
        aggre_net = AggregationNetwork(feature_dims=feature_dims, projection_dim=args.PROJ_DIM, device=device, feat_map_dropout=args.FEAT_MAP_DROPOUT,
                                       num_blocks=args.NUM_BLOCKS, bottleneck_channels=None,)

        if args.LOAD is not None:
            if os.path.exists(args.LOAD): 
                pretrained_dict = torch.load(args.LOAD, map_location=device)
                aggre_net.load_pretrained_weights(pretrained_dict)
                logger.info(f'Load model from {args.LOAD}')
            elif (not args.DO_EVAL) and (not os.path.exists(args.LOAD)): 
                logger.warning(f'Model path {args.LOAD} does not exist. Do not load the model.')
            else:
                ckpt_path = args.LOAD.replace('best.pth', 'last.pth')
                if os.path.exists(ckpt_path):
                    pretrained_dict = torch.load(ckpt_path)
                    aggre_net.load_pretrained_weights(pretrained_dict)
                    logger.info(f'Load model from {ckpt_path}')
                else:
                    logger.warning(f'Model path {args.LOAD} does not exist. Do not load the model.')
                    raise FileNotFoundError(f'Model path {args.LOAD} does not exist.')

    aggre_net.to(device)
    total_args = aggre_net.parameters()
    if args.DENSE_OBJ>0:
        corr_map_net = Correlation2Displacement(setting=args.DENSE_OBJ, window_size=args.SOFT_TRAIN_WINDOW).to(device)
        total_args = chain(total_args, corr_map_net.parameters())
    else:
        corr_map_net = None

  
    if args.DO_EVAL: # eval on test set
        with torch.no_grad():
            pck_010, pck_005, pck_001, result = eval(args, aggre_net, save_path, split='test')
            d_res = {'result': result}
            d_res['pck_010'] = pck_010
            d_res['pck_005'] = pck_005
            d_res['pck_001'] = pck_001
            d_res['args'] = args
            with open(save_path+f'/{results_name}.pkl', 'wb') as f:
                pickle.dump(d_res, f)
    else: 
        optimizer = torch.optim.AdamW(total_args, lr=args.LR, weight_decay=args.WD)
        if args.SCHEDULER is not None:
            if args.SCHEDULER == 'one_cycle':
                scheduler = OneCycleLR(optimizer, max_lr=args.LR, total_steps=args.MAX_STEPS, pct_start=args.SCHEDULER_P1)
                print(f'Using one cycle scheduler with max_lr={args.LR}, total_steps={args.MAX_STEPS}, pct_start={args.SCHEDULER_P1}')
        else:
            scheduler = None
        train(args, aggre_net, corr_map_net, optimizer, scheduler, logger, save_path)

if __name__ == '__main__':
    parser = configargparse.ArgParser()

    # load config
    parser.add_argument(
        '-c', '--config',
        is_config_file=True,
        help='YAML config file path'
    )

    # basic training setting
    parser.add_argument('--EXP_ID', type=int, default=0)                            # experiment id
    parser.add_argument('--SEED', type=int, default=42)                             # random seed
    parser.add_argument('--NOTE', type=str, default='')                             # note for the experiment
    parser.add_argument('--SAMPLE', type=int, default=0)                            # sample 100 pairs for each category for training, set to 0 to use all pairs
    parser.add_argument('--TEST_SAMPLE', type=int, default=100)                     # sample 20 pairs for each category for testing, set to 0 to use all pairs
    parser.add_argument('--TOTAL_SAVE_RESULT', type=int, default=0)                 # save the qualitative results for the first 5 pairs
    parser.add_argument('--IMG_THRESHOLD', action='store_true', default=False)      # set the pck threshold to the image size rather than the bbox size
    parser.add_argument('--ANNO_SIZE', type=int, default=840)                       # image size for the annotation input
    parser.add_argument('--LR', type=float, default=5e-3)                        # learning rate
    parser.add_argument('--WD', type=float, default=1e-3)                           # weight decay
    parser.add_argument('--BZ', type=int, default=1)                                # batch size
    parser.add_argument('--SCHEDULER', type=str, default="one_cycle")               # set to use lr scheduler, one_cycle, None
    parser.add_argument('--SCHEDULER_P1', type=float, default=0.3)                  # set the first parameter for the scheduler
    parser.add_argument('--EPOCH', type=int, default=1)                             # number of epochs
    parser.add_argument('--MAX_STEPS', type=int, default=200_001)               # max number of steps
    parser.add_argument('--EVAL_EPOCH', type=int, default=25_000)                   # number of steps for evaluation
    parser.add_argument('--NOT_WANDB', action='store_true', default=False)          # set true to not use wandb
    parser.add_argument('--MAX_KPS', type=int, default=50)                          # max number of keypoints 
    parser.add_argument('--NOT_VAL', action='store_true', default=False)            # if true, no validation on the own dataset is performed
    parser.add_argument('--NUM_WORKERS', type=int, default=4)                       # max number of keypoints 

    # data
    parser.add_argument('--TRAIN_DATASET', type=str, default='spair_pseudo_gt')     # set the training dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'ap10k' for AP10k, 'spair_pseudo_gt' for SPair-71k (pseudo-labels)
    parser.add_argument('--TRAIN_DATASET_ANN_DIR', type=str, default='')            # Dir of pseudo-label annotations
    parser.add_argument('--IMG_DIR', type=str, default='')                          # directory of images (if not standard)
    parser.add_argument('--FEATURE_DIR', type=str, default='')                      # directory of features (if not standard)
    parser.add_argument('--DATASET_DIR_PF_PASCAL', type=str, default='data/PF-dataset-PASCAL')    # dataset directory for PF-Pascal
    parser.add_argument('--DATASET_DIR_AP10K', type=str, default='data/ap-10k')     # dataset directory for AP10k
    parser.add_argument('--DATASET_DIR_SPAIR', type=str, default='data/SPair-71k')  # dataset directory for SPair-71k
    parser.add_argument('--DATASET_DIR_IN3D', type=str, default='data/IN3D')        # dataset directory for IN3D
    parser.add_argument('--DINO_PATH_SUFFIX', type=str, default='_dino')            # suffix for the dino feature path
    parser.add_argument('--SD_PATH_SUFFIX', type=str, default='_sd')                # suffix for the sd feature path

    # training model setup
    parser.add_argument('--LOAD', type=str, default=None)                           # path to load the pretrained model
    parser.add_argument('--DENSE_OBJ', type=int, default=1)                         # set true to use the dense training objective, 1: enable; 0: disable
    parser.add_argument('--GAUSSIAN_AUGMENT', type=float, default=0.1)              # set float to use the gaussian augment, float for std
    parser.add_argument('--FEAT_MAP_DROPOUT', type=float, default=0.2)              # set true to use the dropout for the feat map
    parser.add_argument('--ENSEMBLE', type=int, default=1)                          # set true to use the ensembles of sd feature maps
    parser.add_argument('--PROJ_DIM', type=int, default=768)                        # projection dimension of the post-processor
    parser.add_argument('--NUM_BLOCKS', type=int, default=1)                        # Number of blocks in the aggregation network
    parser.add_argument('--PAIR_AUGMENT', action='store_true', default=False)       # set true to enable pose-aware pair augmentation
    parser.add_argument('--SELF_CONTRAST_WEIGHT', type=float, default=0)            # set true to use the self supervised loss
    parser.add_argument('--SOFT_TRAIN_WINDOW', type=int, default=0)                 # set true to use the window soft argmax during training, default is using standard soft argmax
    parser.add_argument('--ONLY_DINO', action='store_true', default=False)            # if true, use only DINO
    
    # evaluation setup
    parser.add_argument('--DO_EVAL', action='store_true', default=False)            # set true to do the evaluation on test set
    parser.add_argument('--DUMMY_NET', action='store_true', default=False)          # set true to use the dummy net, used for zero-shot setting
    parser.add_argument('--EVAL_DATASET', type=str, default='spair')                # set the evaluation dataset, 'spair' for SPair-71k, 'pascal' for PF-Pascal, 'ap10k' for AP10k
    parser.add_argument('--AP10K_EVAL_SUBSET', type=str, default='intra-species')          # set the test setting for ap10k dataset, `intra-species`, `cross-species`, `cross-family`
    parser.add_argument('--COMPUTE_GEOAWARE_METRICS', action='store_true', default=False)   # set true to use the geo-aware count
    parser.add_argument('--KPT_RESULT', action='store_true', default=False)         # set true to evaluate per kpt result, in the paper, this is used for comparing unsupervised methods, following ASIC
    parser.add_argument('--ADAPT_FLIP', action='store_true', default=False)         # set true to use the flipped images, adaptive flip
    parser.add_argument('--MUTUAL_NN', action='store_true', default=False)          # set true to use the flipped images, adaptive flip, mutual nn as metric
    parser.add_argument('--SOFT_EVAL', action='store_true', default=False)          # set true to use the soft argmax eval
    parser.add_argument('--SOFT_EVAL_WINDOW', type=int, default=7)                  # set true to use the window soft argmax eval, window size is 2*SOFT_EVAL_WINDOW+1, 0 to be standard soft argmax

    args = parser.parse_args()
    
    print("args:\n",args)
    main(args)