"""
Plot the line profiles going from background to particle for a TEM and a synthetic training
image. The CSV data files are obtained from the line profile tool in ImageJ.

@author: Ross Carmichael
"""

import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image
from scipy import ndimage

img_synth = ndimage.rotate(Image.open("synthetic_section.png"), 180)
img_tem = ndimage.rotate(Image.open("tem_section.png"), 180)

plt.style.use("science")


synth_dist, synth_value = np.loadtxt("synthetic_particle_profile.csv", unpack=True, skiprows=1, delimiter=",")
tem_dist, tem_value = np.loadtxt("tem_particle_profile.csv", unpack=True, skiprows=1, delimiter=",")

fig, axs = plt.subplots(2,2, figsize=(5.3,2.67), gridspec_kw={"width_ratios": [3,1]})

axs[0,0].plot(synth_dist, synth_value, label="synthetic", color="k", linewidth=1, alpha=0.5)
axs[1,0].plot(tem_dist, tem_value, label="TEM", color="k", linewidth=1, alpha=0.5)

axs[0,1].imshow(img_synth, cmap=plt.cm.gray)
axs[1,1].imshow(img_tem, cmap=plt.cm.gray)

axs[1,0].set_xlabel("distance (pixels)")

axs[0,0].set_xlim(0, np.minimum(np.max(tem_dist), np.max(synth_dist)))
axs[1,0].set_xlim(0, np.minimum(np.max(tem_dist), np.max(synth_dist)))

axs[0,1].axis('off')
axs[1,1].axis('off')
axs[0,0].set_xticklabels('')

axs[0,0].legend()
axs[1,0].legend()

# axs[0,0].set_ylabel("gray value")
# axs[1,0].set_ylabel("gray value")

fig.text(0.03, 0.5, "gray value", va="center", rotation="vertical")

plt.savefig("profile_w_image.pdf")
plt.show()