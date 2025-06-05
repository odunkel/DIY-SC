import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple
from matplotlib.patches import ConnectionPatch

def draw_correspondences_gathered(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]],
                        image1: Image.Image, image2: Image.Image, threshold=None) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :return: a figure of images with marked points.
    """
    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)

    if threshold is not None:
        radius3 = threshold[0,0].item()
    else:
        radius3 = 0.03*max(image1.size)
    
    # plot a subfigure put image1 in the top, image2 in the bottom
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.subplots_adjust(wspace=0.025)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    for point1, point2, color in zip(points1, points2, colors):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=color, edgecolor='white', alpha=0.5)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=color, edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius3, facecolor=color, edgecolor='white', alpha=0.3)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=color, edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

    return fig

def add_transparency(image: Image.Image, transparency: float) -> Image.Image:
    """
    Adds transparency to an image.
    :param image: PIL image object.
    :param transparency: Float representing the level of transparency (0.0 to 1.0, where 1.0 is fully opaque).
    :return: New PIL image object with transparency.
    """
    # Ensure the image has an alpha channel
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Prepare the transparency layer
    alpha = int(255 * transparency)
    alpha_layer = Image.new('L', image.size, alpha)

    # Combine the alpha layer with the image
    image.putalpha(alpha_layer)

    return image

def draw_correspondences_lines(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], 
                               gt_points2: List[Tuple[float, float]], image1: Image.Image, 
                               image2: Image.Image, threshold=None, geo_idx=None, geo_err=None, transparency=1,
                               plot_outer_circle=True) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param points2: a list of (y, x) coordinates of image2, corresponding to points1.
    :param gt_points2: a list of ground truth (y, x) coordinates of image2.
    :param image1: a PIL image.
    :param image2: a PIL image.
    :param threshold: distance threshold to determine correct matches.
    :return: a figure of images with marked points and lines between them showing correspondence.
    """
    bias = 0
    points2=points2.cpu().float().numpy()
    gt_points2=gt_points2.cpu().float().numpy()

    no_gt = True if (points2 == gt_points2).all() else False
    if transparency < 1 and transparency > 0:
        image1 = add_transparency(image1, transparency)
        image2 = add_transparency(image2, transparency)

    def compute_correct(threshold):
        alpha = torch.tensor([0.1, 0.05, 0.01])
        correct = torch.zeros(len(alpha))
        err = (torch.tensor(points2) - torch.tensor(gt_points2)).norm(dim=-1)
        err = err.unsqueeze(0).repeat(len(alpha), 1)
        if threshold is None:
            threshold = image1.size[0]
            threshold = torch.tensor([threshold]).unsqueeze(0).repeat(len(alpha), 1)
        else:
            threshold = threshold[:,:err.shape[1]]
        correct = err < threshold.unsqueeze(-1) if len(threshold.shape)==1 else err < threshold
        return correct, err[0]

    correct, err = compute_correct(threshold)
    correct = correct[0] #pck10

    assert len(points1) == len(points2), f"points lengths are incompatible: {len(points1)} != {len(points2)}."
    num_points = len(points1)

    if num_points > 11:
        cmap = plt.get_cmap('inferno')
    else:
        cmap = ListedColormap(["yellow", "blue", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    # colors = np.array([cmap(x) for x in range(num_points)])
    colors = np.array([cmap(x/num_points) for x in range(num_points)])

    radius1, radius2 = 0.05*max(image1.size), 0.01*max(image1.size)
    if plot_outer_circle: radius1 = 0.01*max(image1.size)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.025)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_xlim(0, image1.size[0])
    ax1.set_ylim(image1.size[1], 0)
    ax2.set_xlim(0, image2.size[0])
    ax2.set_ylim(image2.size[1], 0)

    for i, (point1, point2) in enumerate(zip(points1, points2)):
        y1, x1 = point1
        circ1_1 = plt.Circle((x1, y1), radius1, facecolor=colors[i], edgecolor='white', alpha=0.2)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=colors[i], edgecolor='white')
        ax1.add_patch(circ1_1)
        ax1.add_patch(circ1_2)
        y2, x2 = point2
        circ2_1 = plt.Circle((x2, y2), radius1, facecolor=colors[i], edgecolor='white', alpha=0.2)
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=colors[i], edgecolor='white')
        ax2.add_patch(circ2_1)
        ax2.add_patch(circ2_2)

        # Draw lines
        color = '#00FF00' if correct[i].item() else '#FF0000'
        if type(geo_idx) == list:
            if geo_idx is not None and i in geo_idx:
                if color == '#00FF00':
                    if err[i] < geo_err[geo_idx.index(i)]: # if the error to gt is smaller than the geometric error
                        bias += 1
                    else:
                        color = '#FFFF00'
                else:
                    color = '#FFFF00'
        
        color = colors[i] if no_gt else color

        con = ConnectionPatch(xyA=(x2, y2), xyB=(x1, y1), coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, color=color, linewidth=1)
        ax2.add_artist(con)

    return fig, bias



def draw_keypoints_on_img(points1: List[Tuple[float, float]], image1: Image.Image, threshold=None, geo_idx=None, geo_err=None, transparency=1) -> plt.Figure:
    """
    draw point correspondences on images.
    :param points1: a list of (y, x) coordinates of image1, corresponding to points2.
    :param image1: a PIL image.
    :param threshold: distance threshold to determine correct matches.
    :return: a figure of images with marked points and lines between them showing correspondence.
    """
    bias = 0

    vis = points1[:, 1].to(bool)

    num_points = len(points1)

    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x) for x in range(num_points)])
    radius1, radius2 = 0.03*max(image1.size), 0.01*max(image1.size)
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 8))
    # plt.subplots_adjust(wspace=0.025)
    ax1.axis('off')
    ax1.imshow(image1)
    ax1.set_xlim(0, image1.size[0])
    ax1.set_ylim(image1.size[1], 0)

    for i, point1 in enumerate(points1):
        if not vis[i]:
            continue
        y1, x1 = point1
        # circ1_1 = plt.Circle((x1, y1), radius1, facecolor=colors[i], edgecolor='white', alpha=0.5)
        # ax1.add_patch(circ1_1)
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=colors[i], edgecolor='white')
        ax1.add_patch(circ1_2)
        ax1.text(x1 + 15, y1, str(i), color='white', fontsize=15, ha='left', va='center') 
    return fig, bias


def save_visualization(thresholds, pair_idx, vis, save_path, category, 
                       img1_kps, img1, img2, kps_1_to_2, img2_kps, anno_size, adapt_flip=False, transparency=0.75):
    """
    Save visualization of keypoints and their correspondences, including flipped versions if applicable.

    Parameters:
    - thresholds: Thresholds for determining visibility.
    - pair_idx: Index of the current pair being processed.
    - vis: Visibility array for keypoints.
    - save_path: Base path to save the results.
    - category: Category of the current images.
    - img1_kps: Keypoints for image 1.
    - img1: Image 1.
    - img2: Image 2.
    - kps_1_to_2: Correspondences from image 1 to image 2 keypoints.
    - img2_kps: Keypoints for image 2.
    - anno_size: Annotation size used for determining threshold.
    - adapt_flip: Whether to adapt flip based on distance metrics.
    - transparency: Transparency for drawing correspondences.
    """
    tmp_alpha = torch.tensor([0.1, 0.05, 0.01])
    if thresholds is not None:
        tmp_bbox_size = thresholds[pair_idx].repeat(vis.sum()).cpu()
        tmp_threshold = tmp_alpha.unsqueeze(-1) * tmp_bbox_size.unsqueeze(0)
    else:
        tmp_threshold = (tmp_alpha * anno_size).cpu().unsqueeze(-1).repeat(1, vis.sum())

    category_path = os.path.join(save_path, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    if adapt_flip:
        img1_kps[~vis] = 0
        fig, _ = draw_correspondences_lines(img1_kps[vis][:, [1, 0]], kps_1_to_2[vis][:, [1, 0]], img2_kps[vis][:, [1, 0]], img1, img2, tmp_threshold, transparency)
        fig.savefig(os.path.join(category_path, f'{pair_idx}_pred_flip.png'))
    else:
        fig, _ = draw_correspondences_lines(img1_kps[vis][:, [1, 0]], kps_1_to_2[vis][:, [1, 0]], img2_kps[vis][:, [1, 0]], img1, img2, tmp_threshold, transparency)
        fig.savefig(os.path.join(category_path, f'{pair_idx}_pred.png'))

    fig_gt = draw_correspondences_gathered(img1_kps[vis][:, [1, 0]], img2_kps[vis][:, [1, 0]], img1, img2, threshold=tmp_threshold)
    fig_gt.savefig(os.path.join(category_path, f'{pair_idx}_gt.png'))
    plt.close(fig)
    plt.close(fig_gt)

def draw_keypoints(points1: List[Tuple[float, float]], points2: List[Tuple[float, float]], 
                               image1: Image.Image, image2: Image.Image) -> plt.Figure:
    
    num_points = max(len(points1),len(points2))

    if num_points > 4:
        cmap = plt.get_cmap('inferno')
    else:
        cmap = ListedColormap(["red", "yellow", "blue", "lime", "magenta", "indigo", "orange", "cyan", "darkgreen",
                            "maroon", "black", "white", "chocolate", "gray", "blueviolet"])
    colors = np.array([cmap(x/num_points) for x in range(num_points)])

    fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(8, 4))
    plt.subplots_adjust(wspace=0.025)
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)
    ax1.set_xlim(0, image1.size[0])
    ax1.set_ylim(image1.size[1], 0)
    ax2.set_xlim(0, image2.size[0])
    ax2.set_ylim(image2.size[1], 0)

    radius1 = 0.05*max(image1.size)
    radius2 = 0.01*max(image1.size)

    for i, p in enumerate(points1):
        y1, x1 = p
        circ1_2 = plt.Circle((x1, y1), radius2, facecolor=colors[i], edgecolor='white')
        ax1.add_patch(circ1_2)
    
    for i, p in enumerate(points2):
        y2, x2 = p
        circ2_2 = plt.Circle((x2, y2), radius2, facecolor=colors[i], edgecolor='white')
        ax2.add_patch(circ2_2)

    return fig