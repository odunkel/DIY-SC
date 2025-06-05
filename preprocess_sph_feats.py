import argparse
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model_utils.dino_mapper import DINOMapper


class SPairDataset(Dataset):
    def __init__(self, spair_dir, feature_dir):
        self.files = glob.glob(os.path.join(spair_dir, "**", "*.jpg"), recursive=True)
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        dino_feat_path = img_path.replace("JPEGImages", self.feature_dir).replace(".jpg", "_dino.pt")
        img1_dino_load = torch.load(dino_feat_path).detach().squeeze()
        return img1_dino_load, img_path


def main(args):
    # Initialize spherical mapper
    sph_mapper = DINOMapper(backbone='dinov2_vitb14',load_prototypes=False)
    sph_mapper.load_checkpoint(args.ckpt_path, device=args.device, with_prototypes=False)
    sph_mapper = sph_mapper.to(args.device)

    # Create dataset and dataloader
    dataset = SPairDataset(args.spair_dir, args.feature_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Process images
    for batch in tqdm(dataloader):
        img1_dinos, img_paths = batch

        with torch.no_grad():
            dino_feat = img1_dinos.reshape(img1_dinos.shape[0], 768, -1).permute(0, 2, 1).to(args.device)
            img1_sph_raw = sph_mapper.sphere_mapper(dino_feat)
            img1_sph = sph_mapper.post_sph_map(img1_sph_raw)

        for sph, img_path in zip(img1_sph, img_paths):
            sph_path = img_path.replace("JPEGImages", args.feature_dir).replace(".jpg", "_sph.pt")
            torch.save(sph.cpu(), sph_path)
    
    print('Saved SphMap features.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process SPair-71k images and save spherical features.")
    parser.add_argument("--spair_dir", type=str, required=True, help="Path to SPair-71k JPEGImages directory.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument("--feature_dir", type=str, default="features", help="Directory to save features.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for DataLoader.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()

    main(args)