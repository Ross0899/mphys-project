import os
import sys 

import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import cv2
from MightyMosaic import MightyMosaic
from scipy import ndimage as ndi
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

def load_data(path, tif=False):
    if tif:
        images = sorted(glob(os.path.join(path, "*.tif")))
        return images
    else:
        images = sorted(glob(os.path.join(path, "*.png")))
        return images

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = image / 255.0
    image = image.astype(np.float32)

    return image

# get unique file name 
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H.%M.%S")
fname = "sizes_" + timestampStr + ".csv"

# model_1.3
path = "../preprocessing/Out"

# # model_2.0
# path = "../training_archive/training_v2/particles/"
# # model_2.2
# path = "../training_archive/training_v3/particles/"
# # model_3.0
#path = "../preprocessing/data/augmented/particles/"

# File paths
images = load_data(path, tif=True)

if len(images) == 0:
    print(f"No images found in: {path}")
    sys.exit(1)
print(f"Images: {len(images)}")

synthetic_images = [read_image(image) for image in images]

model = tf.keras.models.load_model('saved_model/model_1_TEST.h5')

# add loop over all images

for synthetic_image in synthetic_images:

    mosaic = MightyMosaic.from_array(synthetic_image, (128,128), overlap_factor=4, fill_mode='reflect')
    prediction = mosaic.apply(model.predict, progress_bar=False)

    fused_prediction = prediction.get_fusion()
    labels = np.argmax(fused_prediction, axis=-1)

    # Watershedding 
    img = labels*127
    edges = np.copy(img)
    edges[edges==127] = 0

    dis = ndi.distance_transform_edt(img)
    ndis = -dis

    from skimage.feature import peak_local_max
    locedge = 3
    locreg  = np.ones((locedge, locedge))
    locmax  = peak_local_max(dis, min_distance=6)   # particle centres near local maxima in distance map of eroded image


    mask = np.zeros(dis.shape, dtype=bool)   # set up mask with same shape as distance map and pre-populate with False
    mask[tuple(locmax.T)] = True   # set mask to True at coordinates of local maxima
    markers, num_markers = ndi.label(mask, structure=None,)   # labels objects in mask, False or 0 is considered background
    print("Number of markers     = {}".format(num_markers))

    # now do watershed with markers
    from skimage.segmentation import watershed
    marker_watershed = watershed(ndis, markers=markers, mask=img, watershed_line=True)   # watershed and label image

    # get region properties
    from skimage.measure import regionprops
    regprop = regionprops(marker_watershed)

    # Loop through regprop and append area of each labeled object to list.
    particles = []
    cutoff_area = 500

    for i in range(len(regprop)):
        area = regprop[i].area
        if area < cutoff_area:
            continue

        yc   = regprop[i].centroid[0]
        xc   = regprop[i].centroid[1]
        
        particles.append([yc,xc,area])

    # check whether correct particles found by drawing particle centers
    from matplotlib.patches import Circle
    from math import sqrt, pi

    # Create a figure. Equal aspect so circles look circular
    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.set_aspect('equal')

    # Show the image
    # ax.imshow(synthetic_images[0], cmap=plt.cm.gray)
    # ax.set_title('Original image with watershedded regions')

    # Now, loop through coord arrays, and create a circle at each x,y pair
    rr = 1
    for item in particles:
        xc = item[1]
        yc = item[0]
        rr = sqrt(item[2]/pi)
        circ = Circle((xc,yc), rr, alpha=1.0, edgecolor='red', fill=False, linewidth=2.0)
        ax.add_patch(circ)

    plt.close()

    # plot histogram of radii
    areas = []
    for item in particles:
        area = item[2]
        areas.append(area)

    # plot image histogram
    areas_array = np.array(areas)

    with open(fname, "a") as f:
        np.savetxt(f, areas_array)
    
    particles = []
    areas = []
    areas_array = []
    

sys.path.append("../preprocessing")
from size_distribution import plot
sys.path.append("../cnn")

plot(fname, areas_file=True)
