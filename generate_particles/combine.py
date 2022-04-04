import os
import cv2
import numpy as np

img_path1 = "../tem_images_to_be_classified/particles/" # colourful
img_path2 = "./data/divided/particles/" # greyscale from apeer

images1 = np.array([])
images2 = np.array([])

for image1, image2 in os.listdir(img_path1), os.listdir(img_path2):  #iterate through each file 

    images1 = np.append(images1, image1)
    images2 = np.append(images2, image2)

#sort filenames
sorted_images = np.sort(images)

print("Reading in images of particles...")
for image in tqdm(sorted_images):
    #print(image)
    
    df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
    #Reset dataframe to blank after each loop.

    input_img = cv2.imread(img_path + image)  #Read images
