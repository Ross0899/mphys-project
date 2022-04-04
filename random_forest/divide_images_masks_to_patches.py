"""
The script divides the images up into user-specified patches (e.g. 128x128).
The read in and save paths are defined below.

@author: Sreenivas Bhattiprolu
"""

from patchify import patchify, unpatchify
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Change as necessay (ensure divisible into 512x512)
patch_size = 512
step = 512

images = np.array([])
masks = np.array([])

image_path = "../tem_training_cropped/images/"
mask_path = "../tem_training_cropped/masks/"

if not os.path.exists('./data/divided/particles'):
    try:
        os.makedirs('./data/divided/particles')
    except:
        raise Exception("Could not make './data/divided/particles' directory")

if not os.path.exists('./data/divided/masks'):
    try:
        os.makedirs('./data/divided/masks')
    except:
        raise Exception("Could not make './data/divided/masks' directory")

image_save_path = "./data/divided/particles/"
mask_save_path = "./data/divided/masks/"

# Read in image locations
for img in os.listdir(image_path):
    if not img.endswith(".tif"):
        continue 
    images = np.sort(np.append(images, img))

# Open images
for img in tqdm(images):
    image = cv2.imread(image_path+img, 0)
    image_patches = patchify(image, (patch_size, patch_size), step=step)

    x, y = image_patches.shape[0], image_patches.shape[1]

    for i in range(x):
        for j in range(y):
            patch = image_patches[i,j]
            cv2.imwrite(image_save_path + str(i) + str(j) + "_" + str(img), patch)
            
# Read in mask locations
for msk in os.listdir(mask_path):
    if not msk.endswith(".tiff"):
        continue 
    masks = np.sort(np.append(masks, msk))

assert len(images) == len(masks), "Image and mask lists are different sizes."

# Open masks 
for msk in tqdm(masks):
    mask = cv2.imread(mask_path+msk, 0)
    mask_patches = patchify(mask, (patch_size, patch_size), step=step)

    x, y = mask_patches.shape[0], mask_patches.shape[1]

    for i in range(x):
        for j in range(y):
            patch = mask_patches[i,j]
            cv2.imwrite(mask_save_path + str(i) + str(j) + "_" + "mask_" + str(msk), patch)
    
