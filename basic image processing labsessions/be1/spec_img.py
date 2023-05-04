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

# Image dégradée
img1 = np.zeros((256, 256), dtype="uint8")
value = 0
for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        img1[i, j] = value
    value += 1

# Image avec cercle
img2 = np.zeros((256, 256), dtype="uint8")
rayon = 80
for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if (i-128)**2 + (j-128)**2 < rayon**2:
            img2[i, j] = 255
        else: 
            img2[i, j] = 0

# Image dégradée avec cercle
img3 = np.zeros((256, 256), dtype="uint8")
rayon = 80
value
for i in range(img3.shape[0]):
    for j in range(img3.shape[1]):
        if (i-128)**2 + (j-128)**2 < rayon**2:
            img3[i, j] = value
        else: 
            img3[i, j] = 0
    value += 1


# Histogram
hist1, hist_centers1 = skiexp.histogram(img1)
hist2, hist_centers2 = skiexp.histogram(img2)
hist3, hist_centers3 = skiexp.histogram(img3)


# Image display
fig1=plt.figure(1)                  
plt.subplot(3,2,1), plt.plot(hist_centers1, hist1, lw=2), plt.title('Image 1 histogram')
plt.subplot(3,2,2), plt.imshow(img1), plt.title('Image dégradée '), plt.axis('off')
plt.subplot(3,2,3), plt.plot(hist_centers2, hist2, lw=2), plt.title('Image 2 histogram')
plt.subplot(3,2,4), plt.imshow(img2), plt.title('Image avec cercle '), plt.axis('off')
plt.subplot(3,2,5), plt.plot(hist_centers3, hist3, lw=2), plt.title('Image 3 histogram')
plt.subplot(3,2,6), plt.imshow(img3), plt.title('Image avec cercle dégradé'), plt.axis('off')

plt.show()