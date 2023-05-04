# -*- coding: utf-8 -*-
"""
Created on Thur Nov 9 2022
@author: tachennf
"""

import numpy as np
import skimage.io as skio
from os.path import join
import scipy.ndimage as scnd
import skimage.filters as skf
from skimage import color

import skimage as sk
import skimage.data as skd
import matplotlib.pyplot as plt

import fonctions_seq4 as f4

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

img = skio.imread("Images/muscle.tif")

print("Loaded image has dimensions:", img.shape)
img = sk.img_as_float(img)

# Noise reduction via Gaussian filtering
fimg1 = skf.gaussian(img, sigma=0.1)
fimg2 = skf.gaussian(img, sigma=0.5)

# Image gradient components Gx and Gy computation via Sobel filtering
gx1, gy1 = f4.gradient_components(fimg1)
gx2, gy2 = f4.gradient_components(fimg2)

# Gradient magnitude and orientation computation
mag1 = f4.gradient_magnitude(gx1, gy1)
ori1 = f4.gradient_orientation(gx1, gy1) 
mag2 = f4.gradient_magnitude(gx2, gy2)
ori2 = f4.gradient_orientation(gx2, gy2)

new_grad_img1 = f4.non_max_suppression(mag1, ori1)
new_grad_img2 = f4.non_max_suppression(mag2, ori2)

# Hysteresis thresholding and connectivity testing
#hyst = f4.hysteresis_thresholding(new_grad_img, 0.1, 0.3)
hyst1 = skf.apply_hysteresis_threshold(new_grad_img2, 0.15, 0.25)
hyst2 = skf.apply_hysteresis_threshold(new_grad_img2, 0.2, 0.3)
hyst3 = skf.apply_hysteresis_threshold(new_grad_img2, 0.15, 0.3)
hyst4 = skf.apply_hysteresis_threshold(new_grad_img1, 0.15, 0.25)

fig1=plt.figure(1)
plt.subplot(1, 2, 1), skio.imshow(hyst1, cmap='gray'), plt.title('sigma=0.1, tl=0.15, th=0.25'), plt.axis('off')
plt.subplot(1, 2, 2), skio.imshow(hyst4, cmap='gray'), plt.title('sigma=0.5, tl=0.15, th=0.25'), plt.axis('off')

fig2=plt.figure(2)
plt.subplot(1, 4, 1), skio.imshow(hyst1, cmap='gray'), plt.title('sigma=0.1, tl=0.15, th=0.25'), plt.axis('off')
plt.subplot(1, 4, 2), skio.imshow(hyst2, cmap='gray'), plt.title('sigma=0.1, tl=0.2, th=0.3'), plt.axis('off')
plt.subplot(1, 4, 3), skio.imshow(hyst3, cmap='gray'), plt.title('sigma=0.1, tl=0.15, th=0.3'), plt.axis('off')
plt.show()


