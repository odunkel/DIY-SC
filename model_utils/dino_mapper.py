import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
Adapted from https://github.com/VICO-UoE/SphericalMaps/blob/main/dino_mapper.py
"""

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class CrossAttention(nn.Module):
    def __init__(self, kv_in_dim, q_in_dim, atten_dim, value_dim, out_dim, num_heads=8, qkv_bias=False, 
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = atten_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.atten_dim = atten_dim
        self.value_dim = value_dim

        self.q = nn.Linear(q_in_dim, atten_dim, bias=qkv_bias)
        self.k = nn.Linear(kv_in_dim, atten_dim, bias=qkv_bias)
        self.v = nn.Linear(kv_in_dim, value_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(value_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        Bx, Nx, Cx = x.shape
        Bc, Nc, Cc = cond.shape

        q = self.q(x).reshape(Bx, Nx, self.num_heads, self.atten_dim  // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(cond).reshape(Bc, Nc, self.num_heads, self.atten_dim  // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(cond).reshape(Bc, Nc, self.num_heads, self.value_dim  // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(Bx, Nx, Cx)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CABlock(nn.Module):
    def __init__(self, bloc_dim, cond_dim, atten_dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(bloc_dim)
        self.attn = CrossAttention(
            kv_in_dim=cond_dim, q_in_dim=bloc_dim, atten_dim=atten_dim, value_dim=bloc_dim, out_dim=bloc_dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(bloc_dim)
        mlp_hidden_dim = int(bloc_dim * mlp_ratio)
        self.mlp = Mlp(in_features=bloc_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, cond, return_attention=False):
        y, attn = self.attn(self.norm1(x), cond)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConditionalPrototypes(nn.Module):
    def __init__(self, bloc_dim, cond_dim, atten_dim, num_heads, depth, in_dim=3, mlp_ratio=4., 
        qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, cond_bottleneck=None):
        super().__init__()


        self.proj = nn.Linear(in_dim, bloc_dim)
        self.blocks = nn.ModuleList([CABlock(bloc_dim, cond_dim, atten_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,drop_path, act_layer, norm_layer)]*depth)
        self.cond = cond_bottleneck is not None
        if self.cond:
            self.cond_proj = Mlp(cond_dim, cond_bottleneck) if cond_bottleneck>0 else nn.Identity()


    def forward(self, x, cond):
        x = self.proj(x)
        cond = self.cond_proj(cond) if self.cond else torch.zeros_like(cond)
        for block in self.blocks:
            x = block(x, cond)
        return x


class DINOMapper(nn.Module):    
    def __init__(self, backbone='dinov2_vitb14', n_cats=18, cond_bottleneck=0,load_prototypes=True, **kwargs):
        super().__init__()

        assert 'dino' in backbone, f"""{backbone} is not a DINO model """
        if 'dinov2' in backbone:
            repo = 'facebookresearch/dinov2'
        else:
            repo = 'facebookresearch/dino'

        self.dino = torch.hub.load(repo, backbone)
        self.patch_size = self.dino.patch_size
        self.embed_dim = self.dino.embed_dim

        pred_dim = kwargs.get('pred_dim', 3)
        self.map_dim = pred_dim
        out_dim = pred_dim
        self.out_dim = out_dim
        
        self.sphere_mapper = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim//2),
            nn.GELU(),
            Block(dim=self.embed_dim//2, num_heads=self.embed_dim//2//64),
            nn.Linear(self.embed_dim//2, out_dim),
            )


        sph_size = 2**10
        self.sphere = nn.Embedding(sph_size, out_dim)
        with torch.no_grad():
            self.sphere.weight = nn.Parameter(F.normalize(torch.randn(sph_size, out_dim)), requires_grad=False)

        head_dim = 64
        self.n_cats = n_cats
        cond_dim = self.n_cats

        if load_prototypes:
            self.prototypes = ConditionalPrototypes(
                bloc_dim=self.embed_dim, cond_dim=cond_dim,
                atten_dim=self.embed_dim, num_heads=self.embed_dim//head_dim, 
                depth=2, in_dim=self.map_dim, cond_bottleneck=cond_bottleneck)
        else:
            self.prototypes = None


    def post_sph_map(self, sph):
        sph = F.normalize(sph, dim=-1)
        return sph


    def forward(self, im, gt_mask, max_layer=None, return_dino_cls_tokens=False, return_conf=False, save_dirs=None, conf_strategy=None):
        dino = self.dino
        max_layer = max_layer or len(dino.blocks)
        x = im
        with torch.no_grad():
            dino_out = dino.forward_features(x)
            x = dino_out["x_prenorm"][:,1:]
            cond = dino_out["x_prenorm"][:,0]

        x = x.contiguous()
        sph = self.sphere_mapper(x)
        sph = F.normalize(sph, dim=-1) * gt_mask

        if return_dino_cls_tokens:
            return x, sph, cond, dino_out['x_norm_clstoken'].softmax(dim=1)
        return x, sph, cond



    def sphere_loss_prototypes_implicit(self, x, sph, cond, normalize_feats=False,r_mean=True, conf=None,):

        b_size, seq_len, tok_size = x.shape
        
        flat_sph = sph.reshape(-1, self.map_dim)

        recons = self.prototypes(sph, cond.unsqueeze(1)).view(-1, tok_size)

        if normalize_feats:
            x_mask = (sph.norm(dim=-1, keepdim=True) != 0).repeat(1,1,tok_size)
            masked_x = torch.masked.masked_tensor(x, x_mask, requires_grad=False)
            mean = masked_x.mean(dim=(0,1)).view(1,1,tok_size)
            var = masked_x.var(dim=(0,1)).view(1,1,tok_size)
            x = x - mean.get_data()
            x = x / var.get_data()

        flat_feat = x.view(-1, tok_size)

        #remove background patches
        mask = flat_sph.norm(dim=-1, keepdim=True) != 0
        masked_feat = flat_feat.masked_select(mask).view(-1,tok_size)
        masked_recons = recons.masked_select(mask).view(-1,tok_size)

        l = 1 - F.cosine_similarity(masked_feat, masked_recons, dim=-1)

        return l.mean() if r_mean else l


    def make_checkpoint(self, path):
        # remove dino weights from state dict
        sd = {k:v for k,v in self.state_dict().items() if not 'dino' in k}
        torch.save(sd, path)

    def load_checkpoint(self, path, device=None, with_prototypes=True):
        state_dict = torch.load(path, map_location=device)
        # state dict surgery
        sph_mapper_dict = {k[14:]:v for k,v in state_dict.items() if 'sphere_mapper' in k}
        proto_dict = {k[11:]:v for k,v in state_dict.items() if 'prototypes' in k}
        self.sphere_mapper.load_state_dict(sph_mapper_dict)
        if with_prototypes:
            self.prototypes.load_state_dict(proto_dict)


def _cosine_dist(x,y):
    return 1 - F.cosine_similarity(x,y,dim=-1)

def _corrected_cosine_dist(x,y):
    dotprod = (x*y).sum(-1)
    corrected_dp =  dotprod.mul(1-1e-3).acos().mul(2).div(np.pi)
    return corrected_dp

def compute_triplets_single_sph(sph, coords, n_triplets, ndim=3):
    n_coords = coords.shape[-1]
    mask = sph.norm(dim=-1, keepdim=True)>.01
    valid_points = sph.masked_select(mask).view(-1,ndim)
    valid_coords = coords.masked_select(mask).view(-1,n_coords)
    n_valid_points = valid_points.size(0)
    if n_valid_points==0:
        return None, None
    random_triplets = torch.Tensor(random.choices(range(n_valid_points), k=3*n_triplets)).to(sph.device).long()
    sph_triplets = valid_points[random_triplets].view(3,n_triplets,ndim)
    im_triplets = valid_coords[random_triplets].view(3,n_triplets,n_coords)
    return sph_triplets, im_triplets


def triplet_distance_loss(sph, fm_shape, correct_sine=True, n_triplets=128, margin=0, ndim=3):
    # sph = (B,H*W,N)

    sph = sph[..., :ndim]
    b_size, seq_len, _ = sph.shape
    h, w = fm_shape
    h_map = torch.linspace(0,1,h, device=sph.device).view(-1,1).repeat(1,w)
    w_map = torch.linspace(0,1,w, device=sph.device).view(1,-1).repeat(h,1)
    flat_h, flat_w = h_map.view(-1), w_map.view(-1)
    coords = torch.stack([flat_h, flat_w], dim=-1)
    im_dists = (coords.unsqueeze(0) - coords.unsqueeze(1)).pow(2).sum(-1)

    # randomly query n_point triplets per map
    sph_triplets, im_triplets = [], []
    for s in sph:
        s_t, i_t = compute_triplets_single_sph(s, coords, n_triplets,ndim=ndim)
        if s_t is not None:
            sph_triplets.append(s_t)
            im_triplets.append(i_t)
    if len(sph_triplets)==0:
        return torch.tensor(0.).to(sph.device)
    sph_triplets = torch.cat(sph_triplets, dim=1)
    im_triplets = torch.cat(im_triplets, dim=1)

    s_a, s_b, s_c = sph_triplets.unbind(0)
    i_a, i_b, i_c = im_triplets.unbind(0)

    # Compute positive and negative based on image distances
    im_dist_a_b = (i_b - i_a).pow(2).sum(-1)
    im_dist_a_c = (i_c - i_a).pow(2).sum(-1)
    comparative_im_dist = im_dist_a_b < im_dist_a_c

    anchor = s_a
    pos = torch.where(comparative_im_dist.unsqueeze(-1), s_b, s_c)
    neg = torch.where(comparative_im_dist.unsqueeze(-1), s_c, s_b)

    distfunc = _corrected_cosine_dist if correct_sine else _cosine_dist
    loss = F.triplet_margin_with_distance_loss(anchor, pos, neg, margin=margin, distance_function=distfunc)

    return loss


def triplet_orientation_loss(sph, fm_shape, correct_sine=True, n_triplets=128, thresh=0, p=2, r_mean=True):
    # sph = (B,H*W,3)

    b_size, seq_len, _ = sph.shape
    h, w = fm_shape
    h_map = torch.linspace(0,1,h, device=sph.device).view(-1,1).repeat(1,w)
    w_map = torch.linspace(0,1,w, device=sph.device).view(1,-1).repeat(h,1)
    flat_h, flat_w = h_map.view(-1), w_map.view(-1)
    coords = torch.stack([flat_h, flat_w], dim=-1)
    oriented_map = (coords.unsqueeze(0) - coords.unsqueeze(1))
    im_dists = (coords.unsqueeze(0) - coords.unsqueeze(1)).pow(2).sum(-1)
    patch_indices = torch.arange(seq_len, device=sph.device).view(1,-1,1)

    # randomly query n_point triplets per map
    sph_triplets, im_triplets = [], []
    for s in sph:
        s_t, i_t = compute_triplets_single_sph(s, patch_indices, n_triplets)
        if s_t is not None:
            sph_triplets.append(s_t)
            im_triplets.append(i_t)
    if len(sph_triplets)==0:
        return torch.tensor(0.).to(sph.device)
    sph_triplets = torch.cat(sph_triplets, dim=1)
    im_triplets = torch.cat(im_triplets, dim=1)

    s_a, s_b, s_c = sph_triplets.unbind(0)
    i_a, i_b, i_c = im_triplets.unbind(0)

    # Compute orientation of patches triplets
    vec_a_b = nn.functional.normalize(oriented_map[i_a,i_b], dim=-1)
    vec_a_c = nn.functional.normalize(oriented_map[i_a,i_c], dim=-1)
    dets_im = torch.det(torch.stack([vec_a_b, vec_a_c], dim=-1))

    #Filter based on orthogonality
    strong_triplets = dets_im.abs() > .7
    dets_im = dets_im.masked_select(strong_triplets).view(-1,1)
    s_a = s_a.masked_select(strong_triplets).view(-1,3)
    s_b = s_b.masked_select(strong_triplets).view(-1,3)
    s_c = s_c.masked_select(strong_triplets).view(-1,3)

    # Compute positive and negative based on image orientations
    orientation = dets_im > 0
    origin = s_a
    dir_1 = torch.where(orientation, s_b, s_c)
    dir_2 = torch.where(orientation, s_c, s_b)

    # Create tangeant sphere triplets
    s_1 = dir_1 - origin
    s_2 = dir_2 - origin
    dotprod_1 = (origin * s_1).sum(dim=-1, keepdim=True)
    proj_1 = origin * dotprod_1
    s_1 = s_1 - proj_1
    dotprod_2 = (origin * s_2).sum(dim=-1, keepdim=True)
    proj_2 = origin * dotprod_2
    s_2 = s_2 - proj_2

    # Compute orientation of sphere triplets:
    # normalize and cross product
    s_1 = nn.functional.normalize(s_1, dim=-1)
    s_2 = nn.functional.normalize(s_2, dim=-1)
    cross = torch.cross(s_1, s_2, dim=-1)


    if correct_sine:
        norm = cross.norm(dim=-1, keepdim=True)
        corrected = norm.mul(1-1e-3).asin().mul(2).div(np.pi)
        cross = F.normalize(cross) * corrected

    dotprod = (cross * origin).sum(dim=-1)
    loss = (thresh - dotprod).clip(min=0).pow(p)

    return loss.mean() if r_mean else loss

def relative_viewpoint_loss(sph, vp, n_bins=8, conf=None):
    # sph: (B,256,3), vp: (B,)

    if conf is not None:
        conf = conf.unsqueeze(-1)
        avg_sph = (sph * conf).sum(dim=1) / conf.sum(dim=1)
    else:
        avg_sph = sph.mean(dim=1)
    avg_map = F.normalize(avg_sph, dim=-1) 

    corr = (avg_map.unsqueeze(1) * avg_map.unsqueeze(0)).sum(-1)
    with torch.no_grad():
        bin_size =  2 * np.pi / n_bins 
        ang  = vp * bin_size
        circ = torch.stack([ang.cos(), ang.sin()], dim=-1)
        agreement = (circ.unsqueeze(0) * circ.unsqueeze(1)).sum(-1).to(corr.device)
    
    vp_loss = nn.functional.mse_loss(corr, agreement)

    return vp_loss

def relative_viewpoint_loss_sph_cor(sph, sph_points):

    avg_sph = sph.mean(dim=1)
    avg_map = F.normalize(avg_sph, dim=-1) 

    corr = (avg_map.unsqueeze(1) * avg_map.unsqueeze(0)).sum(-1)
    with torch.no_grad():
        agreement = (sph_points.unsqueeze(0) * sph_points.unsqueeze(1)).sum(-1).to(corr.device)    
    vp_loss = nn.functional.mse_loss(corr, agreement)

    return vp_loss
