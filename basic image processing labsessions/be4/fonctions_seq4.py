# -*- coding: utf-8 -*-
""" Functions of sequence 4"""

import numpy as np
import skimage.io as skio
from os.path import join
import scipy.ndimage as scnd
import skimage.filters as skf
from skimage import color

import skimage as sk
import skimage.data as skd
import matplotlib.pyplot as plt

                            # EXERCICE 1
# =============================================================================
def gradient_components(img):
    """Returns the gradient components Gx and Gy of the input image img
    Parameters
    ----------
    img : 2D array
        Input image
    """
    gx = skf.sobel_h(img)
    gy = skf.sobel_v(img)
    return gx, gy

def gradient_magnitude(gx, gy):
    return np.sqrt(gx**2 + gy**2)

def gradient_orientation(gx, gy):
    return np.arctan2(gy, gx)*180/np.pi


def edge_pixels(mag, t):
    """Returns a binary image with edge pixels set to 1 and non-edge pixels set to 0
    Parameters
    ----------
    mag : 2D array
        Gradient magnitude image
    t : float between 0 and 1
        Threshold value
    """
    b = np.copy(mag)
    mag_max = np.amax(mag)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if mag[i,j] < t*mag_max:
                b[i,j] = 0
            else:
                b[i,j] = 1
    return b

def hysteresis_thresholding(mag, t_low, t_high):
    """Returns a binary image with edge pixels set to 1 and non-edge pixels set to 0
    Parameters
    ----------
    mag : 2D array
        Gradient magnitude image
    t_low : float between 0 and 1
        Low threshold value
    t_high : float between 0 and 1
        High threshold value
    """
    b = np.copy(mag)
    mag_max = np.amax(mag)
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if mag[i,j] < t_low*mag_max:
                b[i,j] = 0
            elif mag[i,j] > t_high*mag_max:
                b[i,j] = 1
            else:
                if is_connected(b, i, j):
                    b[i,j] = 1
                else:
                    b[i,j] = 0
    return b

def is_connected(img, i, j): 
    """ Returns True if the pixel at position (i,j) in img is connected to an edge pixel (i.e. has a value of 1)"""
    if i == 0 or i == img.shape[0]-1: 
        return False
    if j == 0 or j == img.shape[1]-1:
        return False
    else: 
        if img[i, j]==1 or img[i+1, j]==1 or img[i-1, j]==1 or img[i, j+1]==1 or img[i, j-1]==1 or img[i+1, j+1]==1 or img[i-1, j-1]==1 or img[i+1, j-1]==1 or img[i-1, j+1]==1:
            return True
        else: 
            return False

                            # EXERCICE 2
# =============================================================================

def non_max_suppression(mag, ori):
    """Returns the non-maximum suppression of the input image mag"""
    new_grad_img = np.ones(mag.shape)
    first_neigh = 0
    second_neigh = 0
    for i in range(1, mag.shape[0]-1):
        for j in range(1, mag.shape[1]-1):
            # select the 2 neighbours in the gradient direction
            if (-22.5<=ori[i,j]<22.5) or (157.5<ori[i,j] <=180 or -157.5<ori[i, j]<=-180):
                first_neigh = mag[i, j+1]
                second_neigh = mag[i, j-1]
            if (22.5<=ori[i,j]<67.5) or (-157.5<ori[i,j] <=-112.5):
                first_neigh = mag[i-1, j+1]
                second_neigh = mag[i+1, j-1]
            if (67,5<=ori[i,j]<112.5) or (-112.5<ori[i,j] <=-67.5):
                first_neigh = mag[i-1, j]
                second_neigh = mag[i+1, j]
            if (112.5<=ori[i,j]<157.5) or (-67.5<ori[i,j] <=-22.5):
                first_neigh = mag[i-1, j-1]
                second_neigh = mag[i+1, j+1]
                # then suppress the points with non-maximal grad value in the gradient direction
            if (mag[i,j]>=first_neigh) and (mag[i,j]>=second_neigh): 
                new_grad_img[i,j] = mag[i, j]
            else:
                new_grad_img[i,j] = 0
            
    return new_grad_img