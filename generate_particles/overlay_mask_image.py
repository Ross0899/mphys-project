"""
This script reads in the predicted masks and corresponding 
images from the random forest predictor and overlays the 
two to allow them to be checked qualitativley for any 
obvious errors.

@author: Ross Carmichael
@date: 12/10/21
"""
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
import pickle

images = np.array([])
masks = np.array([])

image_list = []
mask_list = []

image_path = "./data/synthetic/particles/"
mask_path = "./data/synthetic/masks/"

if not os.path.exists('./data/overlays'):
    try:
        os.makedirs('./data/overlays')
    except:
        raise Exception("Could not make './data/overlays/' directory")

# Read in image locations
for image in os.listdir(image_path):
    if not image.endswith(".tif"):
        continue
    images = np.sort(np.append(images, image))

# Open, reshape and append to a dataframe
for image in images:
    img = cv2.imread(image_path + image)
    image_list.append(img)

# Read in mask locations
for mask in os.listdir(mask_path):
    if not mask.endswith(".tif"):
        continue
    masks = np.sort(np.append(masks, mask))

# Open, reshape and append to a dataframe
for mask in masks:
    msk = cv2.imread(mask_path + mask)
    mask_list.append(msk)

# Overlay and save 
for i in range(len(image_list)):
    assert len(images) == len(masks), "Image and mask lists are different sizes."

    plt.imshow(image_list[i], cmap="gray", interpolation='nearest')
    plt.imshow(mask_list[i], alpha=0.3, cmap="viridis", interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join("./data/overlays", images[i]))
