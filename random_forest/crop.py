"""
Crop TEM training images and masks to remove the experimental plate and 
length scale.

@author: Ross Carmichael
"""

import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


img_path = " "
images = np.array([])

mask_path = " "
masks = np.array([])

for image in os.listdir(img_path):
    if not image.endswith(".tif"):
        continue
    images = np.append(images, image)

#sort filenames
sorted_images = np.sort(images)

for image in tqdm(sorted_images):
    input_img = cv2.imread(img_path + image)  
    img = input_img[824:3000, :]

    cv2.imwrite("cropped_img_" + image, img)


for mask in os.listdir(mask_path):
    if not image.endswith(".tif"):
        continue
    masks = np.append(masks, mask)

#sort filenames
sorted_masks = np.sort(masks)

for mask in tqdm(sorted_masks):
    input_mask = cv2.imread(mask_path + mask)  
    msk = input_mask[824:3000, :]

    cv2.imwrite("cropped_msk_" + mask, msk)