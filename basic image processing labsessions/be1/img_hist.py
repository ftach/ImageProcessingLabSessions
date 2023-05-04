# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:22 2020

@author: tachennf
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import skimage.exposure as skiexp 
import matplotlib.pyplot as plt

#All figures closing
plt.close('all')
# matplotlib for gray level images display: fix the colormap and 
# the image figure size
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

#img=skio.imread('Images/pout.tif')

# IMAGE CREATION
img1 = np.zeros((256, 256), dtype="uint8")
img2 = np.zeros((256, 256), dtype="uint8")

for i in range(img1.shape[0]):
    for j in range(img2.shape[1]):
        if i < 128:
            img1[i, j] = 25
            img2[i, j] = 225
        else: 
            img1[i, j] = 225
            img2[i, j] = 25 

# Histogram
hist1, hist_centers1 = skiexp.histogram(img1)
hist2, hist_centers2 = skiexp.histogram(img2)

# Image display
fig1=plt.figure(1)                  
plt.subplot(2,2,1), plt.plot(hist_centers1, hist1, lw=2), plt.title('Image 1 histogram')
plt.subplot(2,2,2), plt.imshow(img1), plt.title('Image 1 ')
plt.subplot(2,2,3), plt.plot(hist_centers2, hist2, lw=2), plt.title('Image 2 histogram')
plt.subplot(2,2,4), plt.imshow(img2), plt.title('Image 2 ')


plt.show()