# -*- coding: utf-8 -*-
"""
Python code for generating simulated image of randomly placed, potentially overlapping particles with random radii.

It is probably better/faster to do this with masks,
but the code below works for the intended purpose and is fast enough

Adaptation of original generate_particles.py script by Job Thijsen to make 
images more representative of real TEM images.

Script adapted from     
    @author: Job Thjissen

By Ross Carmichael
@date: 27/10/21
"""

import os
import sys
import csv
from math import sqrt, pi
import random as rd
import imageio 
import cv2
from skimage import draw
import io
import numpy as np
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from datetime import datetime

#from size_distribution import plot

def create_image_array(xsize, ysize):
    """
    Return numpy array of zeros of size (xsize, ysize)
    """
    array = np.zeros((xsize, ysize), dtype=np.uint8)
    mask = np.zeros((xsize, ysize), dtype=np.uint8)

    return array, mask

def draw_background(array, pix_val_av, mu, sigma):
    """
    Draw the background of the image as average plus nornal noise.
    """
    size = array.shape
    xsize = size[0]
    ysize = size[1]
    for i in range(xsize):
        for j in range(ysize):
            array[i,j] = pix_val_av + np.random.normal(mu, sigma)

    return array

def draw_particle(array, mask, xsize, ysize, pcx, pcy, pr, pix_val_av, mu, sigma):
    """
    Add a particle to the array at (pcx, pcy).
    Particle has an average value plus random uniform noise.
    """
    normal_std = np.sqrt(np.log(1 + (sigma/mu)**2))
    normal_mean = np.log(mu) - normal_std**2 / 2

    for i in range(xsize):
        for j in range(ysize):
            rsq = (i-pcx)**2 + (j-pcy)**2
            if rsq < pr**2:
                array[i,j] = pix_val_av + np.random.lognormal(normal_mean, normal_std)
                mask[i,j] = 127
    
    return array, mask

def draw_particle_edge(mask, particle_list):
    """
    Draw the particle edge boundaries around the particle mask.
    """

    stroke = 3
    for item in particle_list:
        for delta in range(-(stroke // 2) + (stroke % 2), (stroke + 1) // 2):
            xc, yc, rr = item
            rr, cc, _ = draw.circle_perimeter_aa(xc, yc, radius=rr+delta, shape=mask.shape)
            mask[rr, cc] = 255

    return mask

def create_first_particle(array, mask, rad_min, rad_max, pix_val_av, mu, sigma, particle_list):
    """
    Add the first particle to the array and return it.
    Particle has a radius between rad_min and rad_max.
    """
    size  = array.shape
    xsize = size[0]
    ysize = size[1]
    xtemp = rd.randint(1, xsize)
    ytemp = rd.randint(1, ysize)

    _mu = 0.5 * (rad_min + rad_max)
    _sigma = 0.5 * (rad_max - rad_min)

    normal_std = np.sqrt(np.log(1 + (_sigma/_mu)**2))
    normal_mean = np.log(_mu) - normal_std**2 / 2

    rtemp = int(np.random.lognormal(normal_mean, normal_std/2))
    particle_list.append([xtemp, ytemp, rtemp])
    array_new, mask_new = draw_particle(array, mask, xsize, ysize, xtemp, ytemp, rtemp, pix_val_av, mu, sigma)

    return array_new, mask_new, particle_list

def add_more_particles(array, mask, rad_min, rad_max, pix_val_av, particle_number, radius_overlap, mu, sigma, particle_list, fname):
    """
    Add particles to the array based on overlap conditions (radius_overlap).
    """
    size  = array.shape
    xsize = size[0]
    ysize = size[1]

    _mu = 0.5 * (rad_min + rad_max)
    _sigma = 0.5 * (rad_max - rad_min)

    normal_std = np.sqrt(np.log(1 + (_sigma/_mu)**2))
    normal_mean = np.log(_mu) - normal_std**2 / 2

    for k in range(particle_number):
        overlap_flag = 0

        xtemp = rd.randint(1, xsize)
        ytemp = rd.randint(1, ysize)
        rtemp = int(np.random.lognormal(normal_mean, normal_std/2))

        for m in range(len(particle_list)):
            min_dist_sq = (1.0 - radius_overlap)*(rtemp + particle_list[m][2])**2
            distsq = (xtemp-particle_list[m][0])**2 + (ytemp-particle_list[m][1])**2
            if distsq <= min_dist_sq:
                overlap_flag = 1

        if overlap_flag:
            continue

        particle_list.append([xtemp, ytemp, rtemp])
        array_new, mask_new = draw_particle(array, mask, xsize, ysize, xtemp, ytemp, rtemp, pix_val_av, mu, sigma)


        with open(fname, "a") as f:
            writer = csv.writer(f)
            writer.writerow((rtemp,))

    return array_new, mask_new, particle_list, fname

def main(number):
    # Filename
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H.%M.%S")
    fname = "sizes_" + timestampStr + ".csv"

    particle_number_options = [100, 10, 1]
    # maximum amount of overlap 2 particles can have
    overlap = 0.1

    # background noise
    mu1 = 30
    sigma1 = 10

    # particle noise
    mu2 = 220
    sigma2 = 5

    # particle min and max radii 
    rmin = 20
    rmax = 40

    if not os.path.exists('./data/synthetic/particles'):
        try:
            os.makedirs('./data/synthetic/particles')
        except:
            raise Exception("Could not make '../data/synthetic/particles/' directory")

    if not os.path.exists('./data/synthetic/masks'):
        try:
            os.makedirs('./data/synthetic/masks')
        except:
            raise Exception("Could not make '../data/synthetic/masks/' directory")
    
    for i in range(number):
        particle_list = []
        particle_number= np.random.choice(particle_number_options)

        print(f"Itteration: {i}")

        # Create empty array & mask
        image_array, mask_array = create_image_array(512, 512)
        
        # Add background to array
        image_array = draw_background(image_array, mu1, mu1, sigma1)
        
        # Add first particle to array & mask
        try:
            image_array, mask_array, particle_list = create_first_particle(image_array, mask_array, rmin, rmax, mu2,  mu2, sigma2, particle_list)
        except: continue
        
        # Add all other particles to array & mask
        try:
            image_array, mask_array, particle_list, fname = add_more_particles(image_array, mask_array, rmin, rmax, 
                                                                        mu2, particle_number, overlap, mu2, sigma2, particle_list, fname)
        except: continue

        # Draw perimeters around particles in mask
        mask_array = draw_particle_edge(mask_array, particle_list)

        # Save images & masks
        filename = './data/synthetic/particles/particles' + str(i) + '.eps'
        try:
            imageio.imwrite(filename, image_array)
        except:
            raise Exception(f"Could not write particles{i}.png to './Out/particles/'")

        filename = './data/synthetic/masks/mask_particles' + str(i) + '.eps'

        try:
            imageio.imwrite(filename, mask_array)
        except:
            raise Exception(f"Could not write mask_particles{i}.png to './Out/masks/'")
    
    #plot(fname)

if __name__== "__main__" :
    try:
        number = int(sys.argv[1])
    except: 
        print(f"\nUsage: python3 {os.path.basename(__file__)} [number of images/masks to generate]\n")
        sys.exit(1)

    main(number)
