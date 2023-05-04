# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 11:24:57 2020

@author: capliera
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.exposure as ske
from os.path import join

def represent_histogram(img_array, hist_legend):
    '''Display an histogram using a image array. '''
    hist, bins = np.histogram(img_array, 256, [0, 256])
    hist = np.append(hist, 0)
    plt.plot(bins, hist, label=hist_legend)
    #plt.legend(loc='upper right')

#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

#Image loadind and display -Histogram computation
# Plot in the same figure the image and its histogram

I = skio.imread('Images/Hawkes_Bay.jpg')

fig1 = plt.figure(1)
plt.subplot(2, 2, 1), skio.imshow(I), plt.title('Hawkes Bay'), plt.axis('off')
plt.subplot(2, 2, 2), represent_histogram(I, 'Hawkes Bay')

#Histogram stretching: use the rescale_intensity() function from scikit-image.exposure
J = I.copy()
J = ske.rescale_intensity(J)

# Plot on the same figure the rescaled image and its histogram
#plt.subplot(2, 2, 3), skio.imshow(J), plt.title('Hawkes Bay rescaled'), plt.axis('off')
#plt.subplot(2, 2, 4), represent_histogram(J, 'Hawkes Bay')
#
#plt.show()

# Histogram equalization: use the equalize_hist() function from the 
# scikit-image.exposure
# Plot on the same figure the equalized image and its histogram
# WARNING the equalized histo has abscissa levels between 0 and 1 => multiply by 256 
# and apply the floor() operation before the hispogram display

K = I.copy()
K = ske.equalize_hist(K)

plt.subplot(2, 2, 3), skio.imshow(K), plt.title('Hawkes Bay equalized'), plt.axis('off')
plt.subplot(2, 2, 4), represent_histogram(K*256, 'Hawkes Bay')

plt.show()

