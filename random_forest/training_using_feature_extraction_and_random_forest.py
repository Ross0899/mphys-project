
# https://youtu.be/-u8PHmHxJ5Q
"""
Feature based segmentation using Random Forest
Demonstration using multiple training images

STEP 1: READ TRAINING IMAGES AND EXTRACT FEATURES 

STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME

STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)

STEP 4: DEFINE THE CLASSIFIER AND FIT THE MODEL USING TRAINING DATA

STEP 5: CHECK ACCURACY OF THE MODEL

STEP 6: SAVE MODEL FOR FUTURE USE

STEP 7: MAKE PREDICTION ON NEW IMAGES

"""
# Change to suit
save_model = False

import numpy as np
import cv2
import pandas as pd
from PIL import Image
import pickle
from matplotlib import pyplot as plt
import os
from yellowbrick.classifier import ROCAUC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from tqdm import tqdm

plt.style.use("science")

####################################################################
## STEP 1:   READ TRAINING IMAGES AND EXTRACT FEATURES 
################################################################
image_dataset = pd.DataFrame()  #Dataframe to capture image features
images = np.array([])

img_path = "./data/divided/particles/"
for image in os.listdir(img_path):  #iterate through each file 
    if not image.endswith(".tif"):
        continue
    images = np.append(images, image)

#sort filenames
sorted_images = np.sort(images)

print("Reading in images of particles...")
for image in tqdm(sorted_images):
    #print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    #Reset dataframe to blank after each loop.

    input_img = cv2.imread(img_path + image)  #Read images

    #Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

################################################################
#START ADDING DATA TO THE DATAFRAME
            
    #Add pixel values to the data frame
    pixel_values = img.reshape(-1)
    df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
    df['Image_Name'] = image   #Capture image name as we read multiple images

# #########################  
# #Generate Gabor features
#     num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
#     kernels = []
#     for theta in range(2): 
#         theta = theta / 4. * np.pi
#         for sigma in (1, 3): 
#             for lamda in np.arange(0, np.pi, np.pi / 4):   #Range of wavelengths
#                 for gamma in (0.05, 0.5): 
                
#                     gabor_label = 'Gabor' + str(num) 
#                     ksize=9
#                     kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
#                     kernels.append(kernel)
#                     #Now filter the image and add values to a new column 
#                     fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
#                     filtered_img = fimg.reshape(-1)
#                     df[gabor_label] = filtered_img 
#                     num += 1 
                    
    
# #####################################################
# #Gerate OTHER FEATURES and add them to the data frame
            
#     #CANNY EDGE
#     edges = cv2.Canny(img, 100,200)   #Image, min and max values
#     edges1 = edges.reshape(-1)
#     df['Canny Edge'] = edges1 #Add column to original dataframe
    
#     #ROBERTS EDGE
#     edge_roberts = roberts(img)
#     edge_roberts1 = edge_roberts.reshape(-1)
#     df['Roberts'] = edge_roberts1
    
#     #SOBEL
#     edge_sobel = sobel(img)
#     edge_sobel1 = edge_sobel.reshape(-1)
#     df['Sobel'] = edge_sobel1
    
#     #SCHARR
#     edge_scharr = scharr(img)
#     edge_scharr1 = edge_scharr.reshape(-1)
#     df['Scharr'] = edge_scharr1
    
#     #PREWITT
#     edge_prewitt = prewitt(img)
#     edge_prewitt1 = edge_prewitt.reshape(-1)
#     df['Prewitt'] = edge_prewitt1
    
#     #GAUSSIAN with sigma=3
#     gaussian_img = nd.gaussian_filter(img, sigma=3)
#     gaussian_img1 = gaussian_img.reshape(-1)
#     df['Gaussian s3'] = gaussian_img1
    
#     #GAUSSIAN with sigma=7
#     gaussian_img2 = nd.gaussian_filter(img, sigma=7)
#     gaussian_img3 = gaussian_img2.reshape(-1)
#     df['Gaussian s7'] = gaussian_img3
    
#     #MEDIAN with sigma=3
#     median_img = nd.median_filter(img, size=3)
#     median_img1 = median_img.reshape(-1)
#     df['Median s3'] = median_img1


######################################                    
#Update dataframe for images to include details for each image in the loop
    image_dataset = image_dataset.append(df)

###########################################################
# STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
    # WITH LABEL VALUES AND LABEL FILE NAMES
##########################################################
mask_dataset = pd.DataFrame()  #Create dataframe to capture mask info.
masks = np.array([])

mask_path = "./data/divided/masks/"    
for mask in os.listdir(mask_path):  #iterate through each file to perform some action
    if not mask.endswith(".tiff"):
        continue
    masks = np.append(masks, mask)

#sort filenames
sorted_masks = np.sort(masks)

print("\nReading in mask images...")
for mask in tqdm(sorted_masks):
    #print(mask)
    
    df2 = pd.DataFrame()
    input_mask = np.array(Image.open(os.path.join(mask_path, mask)))

    #Check if the input mask is RGB or grey and convert to grey if RGB
    if input_mask.ndim == 3 and input_mask.shape[-1] == 3:
        label = cv2.cvtColor(input_mask,cv2.COLOR_BGR2GRAY)
    elif input_mask.ndim == 2:
        label = input_mask
    else:
        raise Exception("The module works only with grayscale and RGB images!")

    #Add pixel values to the data frame
    label_values = label.reshape(-1)
    df2['Label_Value'] = label_values
    df2['Mask_Name'] = mask

    mask_dataset = mask_dataset.append(df2)  #Update mask dataframe with all the info from each mask

##################################################################
 #  STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)
##################################################################
dataset = pd.concat([image_dataset, mask_dataset], axis=1)    #Concatenate both image and mask datasets

print("\nDataset of image pixel values and mask labels...")
print(dataset)

#If you expect image and mask names to be the same this is where we can perform sanity check
#dataset['Image_Name'].equals(dataset['Mask_Name'])   
#
#If we do not want to include pixels with value 0 
#e.g. Sometimes unlabeled pixels may be given a value 0.
#dataset = dataset[dataset.Label_Value != 0]

# Assign training features to X and labels to Y
# Drop columns that are not relevant for training (non-features)
X = dataset.drop(labels = ["Image_Name", "Mask_Name", "Label_Value"], axis=1) 

#Assign label values to Y (our prediction)
Y = dataset["Label_Value"].values 

#Encode Y values to 0, 1, 2
Y = LabelEncoder().fit_transform(Y)

#Split data into train and test to verify accuracy after fitting the model. 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

######################################################################
# STEP 4: Define the classifier and fit a model with our training data
######################################################################

## Instantiate model with n number of decision trees
print("\nStarting training...")
model = RandomForestClassifier(n_estimators = 50, random_state = 42, verbose=2) #TODO: n_estimators=50
## Train the model on training data
model.fit(X_train, y_train)

#########################
# STEP 5: Accuracy check
#########################

prediction_test = model.predict(X_test)

##Check accuracy on test dataset. 
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
print("Classes in the image are: ", np.unique(Y))

# Save model in pickle format
if save_model:
    model_name = "_tem_training_50est.model"
    pickle.dump(model, open(model_name, 'wb'))

#ROC curve for RF
roc_auc=ROCAUC(model, classes=[0, 1, 2], micro=False, macro=False, title=" ")  #Create object
roc_auc.fit(X_train, y_train)
print(roc_auc.score(X_test, y_test))
plt.title("")
roc_auc.show(outpath="rocauc_model4.pdf")

##To test the model on future datasets:
#loaded_model = pickle.load(open(model_name, 'rb'))





