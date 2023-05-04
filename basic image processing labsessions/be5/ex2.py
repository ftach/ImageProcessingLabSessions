# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:23 2020

@author: capliera
"""

from sklearn import cluster
import skimage.io as skio
from os.path import join
import matplotlib.pyplot as plt
import skimage.morphology as skmorph
import skimage as sk
import numpy as np


# rerun it on leopard because of its sensitivity to initialization
# keep img with first cluster around 0.3 for leopard
# run erosion, dilation, opening, closing on leopard and see what gives best result
img = skio.imread("Images/leopard.jpg")
img = sk.img_as_float(img)

# For question 2.2 implemnt here the computation of Gabor features
# Use the gabor() and the gauss() functions to be imported from skimage.filters module
# TO BE COMPLETED

# k-means clustering of the image
X = img.reshape((-1, 1))
k_means = cluster.KMeans(n_clusters=2) # use 3 clusters for the toyobjects image, for leopard too (dilatation and closing works best)
# 2 for bacteria (dilatation fills the bacteria holes )
# Closing can remove small dark spots (i.e. “pepper”) and connect small bright cracks. This tends to “close” up (dark) gaps between (bright) features.
# Opening can remove small bright spots (i.e. “salt”) and connect small dark cracks. This tends to “open” up (dark) gaps between (bright) features.
k_means.fit(X)

# extract means of each cluster & clustered population
clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = k_means.labels_
print('# of Observations:', X.shape)
print('Clusters Means:', clusters_means)

# Display the clustered image
X_clustered.shape = img.shape
fig = plt.figure(1)
plt.subplot(1,2,1), skio.imshow(img), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(X_clustered), plt.title('Kmean segmentation')

# Test morphological operations
eroded = skmorph.erosion(X_clustered)
dilated = skmorph.dilation(X_clustered)
opened = skmorph.opening(X_clustered)
closed = skmorph.closing(X_clustered)  

# Display the morphological operations
fig = plt.figure(2)
plt.subplot(2, 3, 1), plt.imshow(eroded), plt.title('Eroded')
plt.subplot(2, 3, 2), plt.imshow(dilated), plt.title('Dilated')
plt.subplot(2, 3, 3), plt.imshow(opened), plt.title('Opened')
plt.subplot(2, 3, 4), plt.imshow(closed), plt.title('Closed')
plt.subplot(2, 3, 5), plt.imshow(X_clustered), plt.title('Kmean segmentation')

fig3 = plt.figure(3)
# implement gabor filter 
for k in range(1, 8):
    real, imag = sk.filters.gabor(img, frequency=0.5, theta=k*np.pi/8)
    gabor_img = np.sqrt(real**2 + imag**2)
    gabor_img = sk.filters.gaussian(gabor_img, sigma=3)
    gabor_img = skmorph.dilation(gabor_img)
    plt.subplot(2, 4, k), plt.imshow(gabor_img), plt.title('Gabor filter')
    plt.axis('off')

plt.show()