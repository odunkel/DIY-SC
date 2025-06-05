import torch
from PIL import Image
import matplotlib.pyplot as plt
from utils.utils_correspondence import resize
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork


def main():
    num_patches = 60
    stride = 14
    target_res = num_patches * stride
    ckpt_file = 'ckpts/0300_dino_spair/best.pth'

    aggre_net = AggregationNetwork(feature_dims=[768,], projection_dim=768, device='cuda')
    aggre_net.load_pretrained_weights(torch.load(ckpt_file))
    extractor_vit = ViTExtractor('dinov2_vitb14', stride=stride, device='cuda')

    img1_path = 'assets/bus_1.JPEG'
    img1 = Image.open(img1_path).convert('RGB')
    img1 = resize(img1, target_res=target_res, resize=True, to_pil=True)
    batch1 = extractor_vit.preprocess_pil(img1)

    img2_path = 'assets/bus_2.JPEG'
    img2 = Image.open(img2_path).convert('RGB')
    img2 = resize(img2, target_res=target_res, resize=True, to_pil=True)
    batch2 = extractor_vit.preprocess_pil(img2)


    with torch.no_grad():
        feats1_dino = extractor_vit.extract_descriptors(batch1.cuda(), layer=11, facet='token').permute(0,1,3,2).reshape(1, -1, num_patches, num_patches)
        feats1_ref = aggre_net(feats1_dino).reshape(-1, num_patches**2).permute(1, 0)
        feats1_ref = feats1_ref / (torch.linalg.norm(feats1_ref, dim=-1, keepdim=True) + 1e-8)
        
        feats2_dino = extractor_vit.extract_descriptors(batch2.cuda(), layer=11, facet='token').permute(0,1,3,2).reshape(1, -1, num_patches, num_patches)
        feats2_ref = aggre_net(feats2_dino).reshape(-1, num_patches**2).permute(1, 0)
        feats2_ref = feats2_ref / (torch.linalg.norm(feats2_ref, dim=-1, keepdim=True) + 1e-8)


    sim_ij = torch.matmul(feats1_ref, feats2_ref.permute(1,0)).cpu().numpy()
    example_indx = 30*num_patches + 40
    sim_ij = sim_ij[example_indx].reshape(num_patches, num_patches)

    plt.imshow(sim_ij); plt.colorbar()
    plt.savefig('assets/feat_sim_bus1_bus2.png')

if __name__ == '__main__':
    main()