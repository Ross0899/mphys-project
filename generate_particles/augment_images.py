
"""
@author: Sreenivas Bhattiprolu
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import AffineTransform, warp
from skimage import io, img_as_ubyte
import random
import os
from scipy.ndimage import rotate
import cv2
from tqdm import tqdm

images_to_generate = 20
seed = 42

# Define functions for each operation
# Define seed for random to keep the transformation same for image and mask

# Make sure the order of the spline interpolation is 0, default is 3. 
# With interpolation, the pixel values get messed up.
def rotation(image, seed):
    random.seed(seed)
    angle = random.randint(-180,180)
    r_img = rotate(image, angle, mode='reflect', reshape=False, order=0)
    return r_img

def h_flip(image, seed):
    hflipped_img = np.fliplr(image)
    return  hflipped_img

def v_flip(image, seed):
    vflipped_img = np.flipud(image)
    return vflipped_img

def v_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    vtranslated_img = np.roll(image, n_pixels, axis=0)
    return vtranslated_img

def h_transl(image, seed):
    random.seed(seed)
    n_pixels = random.randint(-64,64)
    htranslated_img = np.roll(image, n_pixels, axis=1)
    return htranslated_img


transformations = {
                    'rotate': rotation,
                    'horizontal flip': h_flip, 
                    'vertical flip': v_flip,
                    'vertical shift': v_transl,
                    'horizontal shift': h_transl
                 } 


transformations_partial = {
                            'horizontal flip': h_flip, 
                            'vertical flip': v_flip,
                        } 

# Paths 
image_path="./data/divided/particles/" 
mask_path = "./data/divided/masks/"

if not os.path.exists('./data/augmented/particles'):
    try:
        os.makedirs('./data/augmented/particles')
    except:
        raise Exception("Could not make './data/augmented/particles' directory")

if not os.path.exists('./data/augmented/masks'):
    try:
        os.makedirs('./data/augmented/masks')
    except:
        raise Exception("Could not make './data/augmented/masks' directory")

img_augmented_path="./data/augmented/particles/" 
msk_augmented_path="./data/augmented/masks/" 

images = np.array([])
masks = np.array([])

image_list = []
mask_list = []

# Read in image locations
for img in os.listdir(image_path):      
    if not img.endswith(".tif"):
        continue 
    images = np.sort(np.append(images, img))

# Open and append images to a list 
for image in images:
    img = cv2.imread(image_path + image)
    image_list.append(img)

# Read in mask locations 
for msk in os.listdir(mask_path):     
    if not msk.endswith(".png"):
        continue 
    masks = np.sort(np.append(masks, msk))

assert len(images) == len(masks), "Image and mask lists are different sizes."

for mask in masks:
    msk = cv2.imread(mask_path + mask)
    mask_list.append(msk)

# Perform transformations on all images and masks
count=0
for image, mask in tqdm(zip(image_list, mask_list)):
    for transform in transformations_partial.values():
        image_transform = transform(image, seed)
        mask_transform = transform(mask, seed)

        # Number of transformations applied (sanity check)
        count+=1

        # Save 
        new_image_path= f"{img_augmented_path}/augmented_image_{count}.png" 
        new_mask_path = f"{msk_augmented_path}/augmented_mask_{count}.png"
        
        cv2.imwrite(new_image_path, image_transform)
        cv2.imwrite(new_mask_path, mask_transform)
        
print(f"Images read: {len(image_list)}")
print(f"Masks read: {len(mask_list)}")
print(f"Transformations applied: {count}")
