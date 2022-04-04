import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#img_path = "../tem_training_data/images/"
img_path = "../tem_images_to_be_classified/tem_images/"
images = np.array([])

mask_path = "../tem_training_data/masks/"
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

    cv2.imwrite("../tem_images_to_be_classified/cropped_tem/" + "cropped_" + image, img)


# for mask in os.listdir(mask_path):
#     if not image.endswith(".tif"):
#         continue
#     masks = np.append(masks, mask)

# #sort filenames
# sorted_masks = np.sort(masks)

# for mask in tqdm(sorted_masks):
#     input_mask = cv2.imread(mask_path + mask)  
#     msk = input_mask[824:3000, :]

#     cv2.imwrite("../tem_training_cropped/masks/" + "cropped_" + mask, msk)