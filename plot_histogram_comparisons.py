"""
Plot the size histribution histograms for the various models and methods.

@author: Ross Carmichael
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('science')

def plot(fname, save_name, areas_file=False, norm=False, area='microns', measure='diameter', CONVERSION=1, label=None, alpha=1.0):
    with open(fname, "r") as f:
        values = [line.strip() for line in f.readlines()]
    
    if not areas_file:
        if measure == 'diameter':
            radii = [0.5*CONVERSION*float(v) for v in values]
           # areas = [np.pi * (CONVERSION*float(r)/2)**2 for r in radii]

        elif measure == 'radius':
            radii = [CONVERSION*float(v) for v in values]
            #areas = [np.pi * (CONVERSION*float(r))**2 for r in radii]
    else:
        radii = [np.sqrt((CONVERSION**2)*float(v)/np.pi) for v in values]
        #areas=[float(r)*CONVERSION**2 for r in radii]

    if not norm:
        n, bins, p = plt.hist(radii, bins=20, label=label, alpha=a)
        plt.ylabel("count")
    else:
        n, bins, p = plt.hist(radii, bins=20, density=True, label=label, alpha=alpha)
        plt.ylabel("probability density")

    if area == 'microns':
        plt.xlabel(r"radius ($\mu m$)")
        #plt.xlabel(r"area ($\mu m^{2}$)")
    elif area == 'pix':
        plt.xlabel(r"radius (pix)")
        #plt.xlabel(r"area ($pix^{2}$)")
    
    
    # plt.savefig(save_name)

    return plt

# Convert pix to um
#CONVERSION = 0.01827

# Ground Truth TEM (ASM306_0000.tif)
#plt = plot(fname="../ground_truth_tem_results.txt", save_name='new_s_distrbn_ground_truth_tem.eps', areas_file=False, norm=True, area='microns', measure='diameter', CONVERSION=0.01827, label='ground truth', alpha=0.8)

# Watershedding alone
#plt = plot(fname="../../watershedding/watershedding_TEM_sizes.txt", save_name='new_s_distrbn_watershedding_tem.eps', areas_file=True, norm=True, area='microns', measure='radius', CONVERSION=0.01827, label='watershedding', alpha=0.8)

# Model 1.3 TEM prediction
#plt = plot(fname="./model_comparisons/TEM/model_1.3_TEM_sizes.txt", save_name='s_distrbn_model_1.3_TEM.eps', areas_file=True, norm=True, area='microns', CONVERSION=0.01827)#, label='predicted', alpha=0.5)

# Model 2.2 TEM prediction
#plt = plot(fname="./model_comparisons/TEM/model_2.2_TEM_sizes.txt", save_name='s_distrbn_model_2.2_TEM.eps', areas_file=True, norm=True, area='microns', CONVERSION=0.01827)#, label='predicted', alpha=0.5)

# Model 3.0 TEM prediction
#plt = plot(fname="./model_comparisons/TEM/larger_model_3.0_TEM_sizes.txt", save_name='new_s_distrbn_model_3.0_TEM.eps', areas_file=True, norm=True, area='microns', CONVERSION=0.01827, label='predicted', alpha=0.8)


# Model 3.0 synthetic actual
#plt = plot(fname="./model_comparisons/Synthetic/model_3.0_actual_sizes.csv", save_name='s_distrbn_model_3.0_synth_actual.eps', areas_file=False, norm=True, area='pix', measure='radius', label='ground truth', alpha=0.8)

# Model 3.0 synthetic predicted
#plt = plot(fname="./model_comparisons/Synthetic/model_3.0_predicted_areas.txt", save_name='s_distrbn_model_3.0_synth_predict.eps', areas_file=True, norm=True, area='pix', label='predicted', alpha=0.8)



plt.legend(loc='best')
plt.savefig("new_s_distrbn_compare_synth_actual.pdf")