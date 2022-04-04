"""
Train a random forest model to segment images.

Script adapted from 
    @author: Sreenivas Bhattiprolu

By Ross Carmichael
"""

# Change to suit
save_model = True

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

            
    #Add pixel values to the data frame
    pixel_values = img.reshape(-1)
    df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
    df['Image_Name'] = image   #Capture image name as we read multiple images

    image_dataset = image_dataset.append(df)

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


## Instantiate model with n number of decision trees
print("\nStarting training...")
model = RandomForestClassifier(n_estimators = 50, random_state = 42, verbose=2) #TODO: n_estimators=50
## Train the model on training data
model.fit(X_train, y_train)

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







