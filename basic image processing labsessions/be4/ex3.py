# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:46:37 2020

@author: capliera
"""

import skimage.io as skio
import numpy as np
from os.path import join
from skimage.transform import hough_line, hough_line_peaks, rotate
import math
from scipy import ndimage
import skimage.filters as skf
import matplotlib.pyplot as plt
import skimage.morphology as skm
from skimage.util import img_as_ubyte, crop


plt.rcParams['image.cmap'] = 'gray'
plt.close('all')


img = skio.imread("Images/insurance_form.jpg")
#img = sk.img_as_float(img)
print("Loaded image has dimensions:", img.shape)
fig1 = plt.figure(1)
plt.subplot(2,2,1), plt.imshow(img), plt.title('Original img '), plt.axis('off')

# Image insurance binarization
threshold = 200 # choisi en regardanr pour quelle valeur cela n'enlevait pas de traits noir
b_img = np.ones(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i, j] > threshold:
            b_img[i, j] = 255
        else:
            b_img[i, j] = 0

plt.subplot(2, 2, 2), plt.imshow(b_img), plt.title('Binary img'), plt.axis('off')

# Hough line detection and image angle rotation computation
hspace, angles, dists = hough_line(img)
hspace, angles, dists = hough_line_peaks(hspace, angles, dists)

good_angle = np.max(angles)

# Image Rotation - After image rotation, it is necessary to have a 
# binary image with values 0 or 255
aligned_img = rotate(b_img, good_angle)
aligned_img = crop(aligned_img, (20, 20))
plt.subplot(2, 2, 3), plt.imshow(aligned_img), plt.title('Aligned img'), plt.axis('off')

# Image line restoration with morphologic operators
# Provide in the variable alligned_img a binary image with 255 for the
# background pixels and 0 for the Table information and lines

# Erode with vertical Structure Elt, then close with vertical SE
#SE = np.ones((35,1),np.uint8)
#imgRotNegErode = skm.erosion(255-aligned_img, SE)
#SE = np.ones((201,1),np.uint8)
#imgRotNegErodeClose = skm.closing(imgRotNegErode,SE)
#for i in range(aligned_img.shape[0]):
#    for j in range(aligned_img.shape[1]):
#        if (imgRotNegErodeClose[i,j]>255-aligned_img[i,j]):
#            aligned_img[i,j] = imgRotNegErodeClose[i,j]

# Erode with horizontal SE, then close with horizontal SE
SE = np.ones((1,49),np.uint8)
imgRotNegErode = skm.erosion(255-aligned_img, SE)
SE = np.ones((1,301),np.uint8)
imgRotNegErodeClose = skm.closing(imgRotNegErode,SE)
for i in range(aligned_img.shape[0]):
    for j in range(aligned_img.shape[1]):
        if (imgRotNegErodeClose[i,j]>255-aligned_img[i,j]):
            aligned_img[i,j] = imgRotNegErodeClose[i,j]
plt.subplot(2, 2, 4), plt.imshow(aligned_img), plt.title('Eroded aligned img'), plt.axis('off')
plt.show()