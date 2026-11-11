"""
Vis utilities. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
import scipy

import matplotlib.pyplot as plt

def visualize_img(image, vis_folder, im_name):
    pltname = f"{vis_folder}/{im_name}"
    Image.fromarray(image).save(pltname)
    print(f"Original image saved at {pltname}.")

def visualize_predictions(img, pred, vis_folder, im_name, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_pred.jpg"
        Image.fromarray(image).save(pltname)
        print(f"Predictions saved at {pltname}.")
    return image


def visualize_mask_overlay(img, foreground, vis_folder, im_name, patch_size=16,
                           alpha=0.7, save=True, interpolation="nearest"):
    """Overlay TokenCut's connected-component mask in red on the input image.

    ``foreground`` is the patch-grid mask returned by :func:`object_discovery.ncut`.
    ``nearest`` is the protocol-faithful rendering used by evaluation artifacts.
    ``bilinear`` is a display-only soft-alpha rendering: it does not change the
    patch mask, box, CorLoc, or any saved evaluation result.
    """
    image = np.asarray(img).copy()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    mask = np.asarray(foreground)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D patch mask, got shape {mask.shape}")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    if interpolation == "nearest":
        # ncut's dims are [padded_height / patch_size, padded_width / patch_size].
        # Nearest expansion reproduces the patch mask without adding post-processing.
        mask_full = np.repeat(np.repeat(mask > 0.5, patch_size, axis=0), patch_size, axis=1)
        mask_full = mask_full[:image.shape[0], :image.shape[1]]
    elif interpolation == "bilinear":
        # Resize the patch-grid probabilities to the padded image grid only for
        # presentation. The result is a soft alpha field, never an evaluation mask.
        padded_height = mask.shape[0] * patch_size
        padded_width = mask.shape[1] * patch_size
        mask_full = cv2.resize(mask.astype(np.float32), (padded_width, padded_height),
                               interpolation=cv2.INTER_LINEAR)
        mask_full = mask_full[:image.shape[0], :image.shape[1]]
        mask_full = np.clip(mask_full, 0.0, 1.0)
    else:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    overlay = image.astype(np.float32)
    red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
    if interpolation == "nearest":
        overlay[mask_full] = (1.0 - alpha) * overlay[mask_full] + alpha * red
    else:
        alpha_map = alpha * mask_full[..., None]
        overlay = (1.0 - alpha_map) * overlay + alpha_map * red
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_mask_overlay.png"
        Image.fromarray(overlay).save(pltname)
        print(f"Mask overlay saved at {pltname}.")
    return overlay
  
def visualize_predictions_gt(img, pred, gt, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    # Plot the ground truth box
    if len(gt>1):
        for i in range(len(gt)):
            cv2.rectangle(
                image,
                (int(gt[i][0]), int(gt[i][1])),
                (int(gt[i][2]), int(gt[i][3])),
                (0, 0, 255), 3,
            )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_BBOX.jpg"
        Image.fromarray(image).save(pltname)
        #print(f"Predictions saved at {pltname}.")
    return image

def visualize_eigvec(eigvec, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the second smallest eigvector
    """
    eigvec = scipy.ndimage.zoom(eigvec, scales, order=0, mode='nearest')
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_attn.jpg"
        plt.imsave(fname=pltname, arr=eigvec, cmap='cividis')
        print(f"Eigen attention saved at {pltname}.")
