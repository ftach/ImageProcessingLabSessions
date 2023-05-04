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
        if i > 100 and i < 156 and j > 100 and j < 156:
            img1[i, j] = 135
            img2[i, j] = 135
        else: 
            img1[i, j] = 100
            img2[i, j] = 170 


# Image display
fig1=plt.figure(1)       
plt.subplot(1,3,1), plt.imshow(img1), plt.title('Image 1 with Pyplot'), plt.axis('off')
plt.subplot(1,3,2), skio.imshow(img1), plt.title('Image 1 with Skimage'), plt.axis('off')
plt.subplot(1,3,3), skio.imshow(img2), plt.title('Image 2 with Skimage'), plt.axis('off')


plt.show()