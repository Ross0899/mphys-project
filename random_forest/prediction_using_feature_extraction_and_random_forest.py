
# https://youtu.be/-u8PHmHxJ5Q
"""
@author: Sreenivas Bhattiprolu
"""
###############################################################
#STEP 7: MAKE PREDICTION ON NEW IMAGES
################################################################ 
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

#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

# #Generate Gabor features
#     num = 1
#     kernels = []
#     for theta in range(2):
#         theta = theta / 4. * np.pi
#         for sigma in (1, 3):
#             for lamda in np.arange(0, np.pi, np.pi / 4):
#                 for gamma in (0.05, 0.5):
#                     gabor_label = 'Gabor' + str(num)
#                     ksize=9
#                     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
#                     kernels.append(kernel)
#                     #Now filter image and add values to new column
#                     fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
#                     filtered_img = fimg.reshape(-1)
#                     df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
#                     num += 1

# ######################################################
# #Geerate OTHER FEATURES and add them to the data frame

#     #Feature 3 is canny edge
#     edges = cv2.Canny(img, 100,200)   #Image, min and max values
#     edges1 = edges.reshape(-1)
#     df['Canny Edge'] = edges1 #Add column to original dataframe

#     #Feature 4 is Roberts edge
#     edge_roberts = roberts(img)
#     edge_roberts1 = edge_roberts.reshape(-1)
#     df['Roberts'] = edge_roberts1

#     #Feature 5 is Sobel
#     edge_sobel = sobel(img)
#     edge_sobel1 = edge_sobel.reshape(-1)
#     df['Sobel'] = edge_sobel1

#     #Feature 6 is Scharr
#     edge_scharr = scharr(img)
#     edge_scharr1 = edge_scharr.reshape(-1)
#     df['Scharr'] = edge_scharr1

#     #Feature 7 is Prewitt
#     edge_prewitt = prewitt(img)
#     edge_prewitt1 = edge_prewitt.reshape(-1)
#     df['Prewitt'] = edge_prewitt1

#     #Feature 8 is Gaussian with sigma=3
#     gaussian_img = nd.gaussian_filter(img, sigma=3)
#     gaussian_img1 = gaussian_img.reshape(-1)
#     df['Gaussian s3'] = gaussian_img1

#     #Feature 9 is Gaussian with sigma=7
#     gaussian_img2 = nd.gaussian_filter(img, sigma=7)
#     gaussian_img3 = gaussian_img2.reshape(-1)
#     df['Gaussian s7'] = gaussian_img3

#     #Feature 10 is Median with sigma=3
#     median_img = nd.median_filter(img, size=3)
#     median_img1 = median_img.reshape(-1)
#     df['Median s3'] = median_img1

    return df

##################################################
#Applying trained model to segment multiple files. 

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
    #im.save('../tem_images_to_be_classified/masks/'+ 'mask_' + image)
    #io.imsave('../tem_images_to_be_classified/masks/'+ 'mask_' + image, segmented)
    #cv2.imwrite('../tem_images_to_be_classified/masks/'+ 'mask_' + image, segmented)
    # plt.imshow(segmented)
    #plt.imsave('../tem_images_to_be_classified/masks/'+ 'mask_' + image[:-4] + ".png", segmented)

    
    
    
    
    
    