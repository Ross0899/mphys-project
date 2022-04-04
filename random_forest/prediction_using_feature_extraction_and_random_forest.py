"""
Make preictions using the random forest trained model.

Script adapted from 
    @author: Sreenivas Bhattiprolu

By Ross Carmichael
"""

import numpy as np
import cv2
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from skimage import io
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from tqdm import tqdm
import os
from PIL import Image
import imageio

def feature_extraction(img):
    df = pd.DataFrame()

    img2 = img.reshape(-1)
    df['Original Image'] = img2

    return df


filename = "tem_training.model"
loaded_model = pickle.load(open(filename, 'rb'))

path = "../tem_images_to_be_classified/divided_tem/"

for image in tqdm(os.listdir(path)):
    if not image.endswith(".tif"):
        continue

    input_img= cv2.imread(path+image)

    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    #Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))

    imageio.imwrite('../tem_images_to_be_classified/masks/'+ 'mask_' + image[:-4] + ".png", segmented)
