import argparse
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.utils_correspondence import resize


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def process_and_save_features(file_paths, sd_size, dino_size, layer, facet, model, aug, extractor_vit, num_ensemble, flip=False, do_sd=False, do_dino=False,add_str=''):
    ii = 0
    for file_path in tqdm(file_paths, desc="Processing images (Flip: {})".format(flip)):
        
        subdir_name = 'features' if num_ensemble == 1 else f'features_ensemble{num_ensemble}'
        output_subdir = file_path.replace('JPEGImages', subdir_name).rsplit('/', 1)[0]
        os.makedirs(output_subdir, exist_ok=True)
        
        suffix = f'{add_str}_flip' if flip else f'{add_str}'
        output_path_dino = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_dino{suffix}.pt')
        output_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_sd{suffix}.pt')

        ii += 1
        img1 = Image.open(file_path).convert('RGB')
        if flip:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img1_input = resize(img1, sd_size, resize=True, to_pil=True)
        img1 = resize(img1, dino_size, resize=True, to_pil=True)

        if do_sd:
            accumulated_features = {}
            for _ in range(num_ensemble): 
                features1 = process_features_and_mask(model, aug, img1_input, mask=False, raw=True)
                del features1['s2']
                for k in features1:
                    accumulated_features[k] = accumulated_features.get(k, 0) + features1[k]

            for k in accumulated_features:
                accumulated_features[k] /= num_ensemble

            output_path = os.path.join(output_subdir, os.path.splitext(os.path.basename(file_path))[0] + f'_sd{suffix}.pt')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            try:
                torch.save(accumulated_features, output_path)
            except:
                print(f"Error saving SD features of {file_path}")

        if do_dino:
            img1_batch = extractor_vit.preprocess_pil(img1)
            with torch.no_grad():
                img1_desc_dino = extractor_vit.extract_descriptors(img1_batch.cuda(), layer, facet).permute(0, 1, 3, 2).reshape(1, -1, 60, 60)
            
            os.makedirs(os.path.dirname(output_path_dino), exist_ok=True)
            try:
                torch.save(img1_desc_dino, output_path_dino)
            except:
                print(f"Error saving DINO features of {file_path}")

if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description="Process and save features from images.")
    parser.add_argument('--base_dir', type=str, default='data/SPair-71k/JPEGImages', help='Base directory containing images.')
    parser.add_argument('--dino', action='store_true', help='Whether to compute DINO features.')
    parser.add_argument('--sd', action='store_true', help='Whether to compute SD features.')
    parser.add_argument('--do_flip', action='store_true', help='Whether to flip images vertically.')
    parser.add_argument('--sd_size', type=int, default=960, help='Image size for SD.')
    parser.add_argument('--dino_size', type=int, default=840, help='Image size for DINOv2.')
    parser.add_argument('--layer', type=int, default=11, help='DINOv2 layer for feature extraction.')
    parser.add_argument('--facet', type=str, default='token', help='Facet for feature extraction.')
    parser.add_argument('--num_ensemble', type=int, default=1, help='Number of ensembles for SD processing.')
    args = parser.parse_args()

    set_seed()

    all_files = sorted([os.path.join(subdir, file) for subdir, dirs, files in os.walk(args.base_dir) for file in files if file.endswith('.jpg') or file.endswith('.JPEG') or file.endswith('.png')])
    print('Number of images', len(all_files))

    # Load models
    model, aug = None, None
    extractor_vit = None
    if args.sd: 
        from model_utils.extractor_sd import load_model, process_features_and_mask
        model, aug = load_model(diffusion_ver='v1-5', image_size=args.sd_size, num_timesteps=50, block_indices=[2, 5, 8, 11])
    if args.dino: 
        from model_utils.extractor_dino import ViTExtractor
        extractor_vit = ViTExtractor('dinov2_vitb14', 14, device='cuda')

    try:
        process_and_save_features(all_files, args.sd_size, args.dino_size, args.layer, args.facet, model, aug, extractor_vit, args.num_ensemble, flip=args.do_flip, do_dino=args.dino, do_sd=args.sd,add_str='')
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    
    print("Feature processing completed.")