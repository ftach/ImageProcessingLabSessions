# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 07:58:47 2020

@author: tachennf
"""

import numpy as np
import skimage.io as skio
from os.path import join
import scipy.ndimage as scnd
import skimage.filters as skf
from skimage import color, img_as_ubyte, img_as_float

import skimage as sk
import skimage.data as skd
import matplotlib.pyplot as plt

# load image
img = skio.imread("photo_identite.jpg")
print("Loaded image has dimensions:", img.shape)
#img = sk.img_as_float(img)
print(type(img[0, 0, 0]))
bw_img = color.rgb2gray(img)
fig1=plt.figure(1)
plt.subplot(1,3,1), plt.imshow(img), plt.title("original image"), plt.axis("off")

#  Luminance computation (if the original image is a color image)
size = img.shape
def luminance(img):
    img_l = np.zeros((size[0], size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            img_l[i, j] = 0.298*img[i, j, 0] + 0.586*img[i, j, 1] + 0.1148*img[i, j, 0]
    return img_l 

img_l = luminance(img)
plt.subplot(1, 3, 2), plt.imshow(img_l), plt.title("luminance image"), plt.axis("off")

#  Image video inversion (I=255-I)
def inverse(img):
    size = img.shape
    i_img = np.zeros((size[0], size[1]))
    for i in range(size[0]):
        for j in range(size[1]):
            i_img[i, j] = 255 - img[i, j]
    return i_img

i_img = inverse(bw_img)
print(type(i_img[0, 0]))

plt.subplot(1, 3, 3), skio.imshow(i_img), plt.title("inverted image"), plt.axis("off")
plt.show()
#  Inverted image blurring with a Gaussian filter
#  Final sketch computation in the following way:
    # sketch = (inverted_blurr_img*255)/inverted_img
# where all the values higher or equal to 255 are casted to 255 
