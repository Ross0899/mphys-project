"""
Given a text file of particle radii in pixels, the area
in pixels^2 is output.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot(fname, areas_file=False):
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H.%M.%S")

    with open(fname, "r") as f:
        lines = f.readlines()
        radii = [line.strip() for line in lines]
    
    if not areas_file:
        areas = [np.pi * float(r)**2 for r in radii]
    else:
        areas=[float(r) for r in radii]

    plt.hist(areas, bins=20)
    plt.xlabel("area (pixels squared)")
    plt.ylabel("count")
    plt.savefig("size_dis_" + timestampStr + ".png")