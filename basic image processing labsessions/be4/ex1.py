# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:58:47 2020

@author: capliera
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
#img = skd.coffee()
#img = color.rgb2gray(img)

print("Loaded image has dimensions:", img.shape)
img = sk.img_as_float(img)


gx, gy = f4.gradient_components(img)

# Gradient magnitude and orientation computation 
mag = f4.gradient_magnitude(gx, gy)
ori = f4.gradient_orientation(gx, gy)

fig1=plt.figure(1)
plt.subplot(1, 2, 1), skio.imshow(mag, cmap='gray'), plt.title('Gradient magnitude')
plt.subplot(1, 2, 2), skio.imshow(ori, cmap='gray'), plt.title('Gradient orientation')

#Edge pixels detection : 1st method
b1 = f4.edge_pixels(mag, 0.15)
b2 = f4.edge_pixels(mag, 0.2)
b3 = f4.edge_pixels(mag, 0.25)
fig2=plt.figure(2)
plt.subplot(1, 4, 1), skio.imshow(img, cmap='gray'), plt.title('Original image'), plt.axis('off')
plt.subplot(1, 4, 2), skio.imshow(b1, cmap='gray'), plt.title('t=0.15'), plt.axis('off')
plt.subplot(1, 4, 3), skio.imshow(b2, cmap='gray'), plt.title('t=0.2'), plt.axis('off')
plt.subplot(1, 4, 4), skio.imshow(b3, cmap='gray'), plt.title('t=0.25'), plt.axis('off')

# Edge pixels detection : second method with hysteresis thresholding

#h1 = f4.hysteresis_thresholding(mag, 0.1, 0.5)
#h2 = f4.hysteresis_thresholding(mag, 0.1, 0.6)
#h3 = f4.hysteresis_thresholding(mag, 0.2, 0.6)
h1 = skf.apply_hysteresis_threshold(mag, 0.1, 0.3)
h2 = skf.apply_hysteresis_threshold(mag, 0.15, 0.3)
h3 = skf.apply_hysteresis_threshold(mag, 0.2, 0.3)
fig3=plt.figure(3)
plt.subplot(1, 4, 1), skio.imshow(img, cmap='gray'), plt.title('Original image'), plt.axis('off')
plt.subplot(1, 4, 2), skio.imshow(h1, cmap='gray'), plt.title('t_low=0.1, t_high=0.3'), plt.axis('off')
plt.subplot(1, 4, 3), skio.imshow(h2, cmap='gray'), plt.title('t_low=0.15, t_high=0.3'), plt.axis('off')
plt.subplot(1, 4, 4), skio.imshow(h3, cmap='gray'), plt.title('t_low=0.2, t_high=0.3'), plt.axis('off')
plt.show()  
