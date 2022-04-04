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

image_path = "../tem_images_to_be_classified/cropped_tem/"

if not os.path.exists('../tem_images_to_be_classified/divided_tem/'):
    try:
        os.makedirs('../tem_images_to_be_classified/divided_tem/')
    except:
        raise Exception("Could not make '../tem_images_to_be_classified/divided_tem/' directory")



image_save_path = "../tem_images_to_be_classified/divided_tem/"

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
            
    
