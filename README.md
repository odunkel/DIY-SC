<h2 align="center">Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels</h2>
<div align="center"> 
  <a href="https://odunkel.github.io" target="_blank">Olaf Dünkel</a>, 
  <a href="https://wimmerth.github.io/" target="_blank">Thomas Wimmer</a>,
  <a href="https://people.mpi-inf.mpg.de/~theobalt" target="_blank">Christian Theobalt</a>,
  <a href="https://chrirupp.github.io/" target="_blank">Christian Rupprecht</a>,
  <a href="https://genintel.mpi-inf.mpg.de/" target="_blank">Adam Kortylewski</a>
</div>


<br>

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://genintel.github.io/DIY-SC)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2506.05312)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://891603b6c63d4e2e4f.gradio.live/)

## 💡 DIY-SC for refining DINOv2 features
DIY-SC improves semantic correspondence by refining foundational features, which has potential applications for tasks that require semantically aware features.
In the following, we present how DIY-SC can be integrated straightforwardly in existing code bases, e.g., as an alternative to DINOv2.

### 🛠️ Setup 
While the adapter only requires `torch`, the demonstration below requires the following:
```
pip install pillow matplotlib
pip install torch torchvision torchaudio
```

### Example usage
DINOv2 features can be refined in the following way:

```
from model_utils.projection_network import AggregationNetwork

# Load model
ckpt_dir = 'ckpts/0300_dino_spair/best.pth'
aggre_net = AggregationNetwork(feature_dims=[768,], projection_dim=768)
aggre_net.load_pretrained_weights(torch.load(ckpt_dir))

# Refine features
desc_dino = <DINOv2 features of torch.Size([1, 768, N, N])>
with torch.no_grad():
    desc_proj = aggre_net(desc_dino) # of shape torch.Size([1, 768, N, N])
```

We show an examplary application in `demo_diy.py`.

## 💡 Semantic correspondence on SPair-71k

In the following, we evaluate DIY-SC on the semantic correspondence benchmark SPair-71k.

We present two options for evaluation, pseudo-label generation, and training of the adapter:

I) **Light-weight (DINOv2)**: This strategy only requires DINOv2 features, which reduces installation and compute requirements.

II) **Full (SD+DINOv2)**: This strategy relies on DINOv2 and SD features and, therefore, requires third library pacakages and is computationally heavier.


### 🛠️ Setup

#### I) Light-weight (DINOv2)
To support pseudo-label generation and SPair-71k experiments, install the following packages:
```
pip install tqdm opencv-python wandb ConfigArgParse loguru pandas scipy
```
To extract instance masks, install [SAM](https://github.com/facebookresearch/segment-anything) and download the checkpoint:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P ckpts
```

#### II) Full (SD+DINOv2)
We refer to [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC?tab=readme-ov-file#environment-setup) for the instructions to install the requirements for computing the SD features.

### Preparing SPair-71k data

Download SPair-71k (as in Geo-Aware) by running `bash scripts/download_spair.sh`.

Compute SAM masks with `bash scripts/compute_sam_masks.sh`.

Pre-compute the feature maps by running:
| Option                | Command                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| I) DINOv2 |`bash scripts/precompute_features.sh --dino`                              |
| II) SD+DINOv2     | `bash scripts/precompute_features.sh --dino --sd` |



### Evaluation
Pre-trained adapters can be evaluated on the SPair-71k test split via:
| Option                | Command                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| I) DINOv2 |`python pck_train.py --config configs/eval_spair.yaml --ONLY_DINO --LOAD ckpts/0300_dino_spair/best.pth`                              |
| II) SD+DINOv2     | `python pck_train.py --config configs/eval_spair.yaml --LOAD ckpts/0280_spair/best.pth` |

### Generation of pseudo-labels
We provide generated pseudo-labels on [GDrive](https://drive.google.com/drive/folders/1nGjNsWpqbcqUJS-fNXU_41pMBMdE42Je?usp=sharing). Download them to `data`.

Alternatively, to generate pseudo-labels, perform the following steps.
First, compute the spherical points:
```
bash scripts/precompute_features.sh --sph
```
Then, generate pseudo-labels for validation and training splits (with the `--only_dino` flag for the light-weight option I with only DINO).
```
TRAIN_DATASET_ANN_DIR = 'data/pair_annotations/spair/v145'
python gen_pseudo_labels.py --filter_sph --subsample 300 --split val --TRAIN_DATASET_ANN_DIR $TRAIN_DATASET_ANN_DIR
python gen_pseudo_labels.py --filter_sph --subsample 30_000 --split trn --TRAIN_DATASET_ANN_DIR $TRAIN_DATASET_ANN_DIR
```

### Training
Finally, the refinement adapter is trained via:
```
python pck_train.py --config configs/train_spair.yaml --EXP_ID 0
```

## Planned features
The following features are currently planned or already in development:

- [ ] `torchhub` integration
- [ ] Checkpoints with varying feature dimensions
- [ ] ImageNet-3D training functionality
- [ ] OrientAnything integration and further scaling
- [ ] LoftUp support

**Contributions welcome**: If you have feature suggestions or would like to help implement any of the above, feel free to open an issue or submit a pull request.

## 🎓 Citation
If you find our work useful, please cite:

```bibtex
@misc{duenkel2025diysc,
    title = {Do It Yourself: Learning Semantic Correspondence from Pseudo-Labels},
    author = {D{\"u}nkel, Olaf and Wimmer, Thomas and Theobalt, Christian and Rupprecht, Christian and Kortylewski, Adam},
    booktitle = {arXiv},
    year = {2025},
}
```

## Acknowledgement
We thank [GeoAware-SC](https://github.com/Junyi42/GeoAware-SC) and [SphericalMaps](https://github.com/VICO-UoE/SphericalMaps) for open-sourcing their great works.


