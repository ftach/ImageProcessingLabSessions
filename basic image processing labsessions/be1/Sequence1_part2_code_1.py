# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:22 2020

@author: capliera
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt

def quantize(im, levels):
    """
    Function to run uniform gray-level gray-scale Quantization.
    This takes in an image, and buckets the gray values depending on the params.
    Args:
        im (array): image to be quantized as an array of values from 0 to 255
        levels (int): number of grey levels to quantize to.          
    Return:
        the quantized image
    """
   
    # get int type
    dtype = im.dtype    
    returnImage = np.floor((im/(256/float(levels))))

    print(returnImage)
    return np.array(returnImage, dtype)

def disp_subplot_img(img, title, nb_rows, nb_cols, pos, axis_off=True):
    """
    Function to display an image in a subplot.
    Args:
        img (array): image to be displayed as an array of values from 0 to 255
        title (str): title of the subplot
        nb_rows (int): number of rows in the subplot
        nb_cols (int): number of columns in the subplot
        pos (int): position of the subplot
    Return:
        None
    """
    plt.subplot(nb_rows, nb_cols, pos)
    plt.imshow(img)
    plt.title(title)
    if axis_off:
        plt.axis('off')

#All figures closing
plt.close('all')
# matplotlib for gray level images display: fix the colormap and 
# the image figure size
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

## Grey level image uniform quantization

I_256=skio.imread('Images/Fruits.bmp')
# print(I_256.shape)
# Image display
fig1=plt.figure(1)                  
skio.imshow(I_256)
plt.subplot(2,4,1), plt.imshow(I_256), plt.title('256 grey levels'), plt.axis('off')

#Image quantization with different numbers of grey levels

I_128 = quantize(I_256, 128)
plt.subplot(2,4,2), plt.imshow(I_128), plt.title('128 grey levels'), plt.axis('off')

I_64 = quantize(I_256, 64)    
plt.subplot(2,4,3), plt.imshow(I_64), plt.title('64 grey levels'), plt.axis('off')

I_32 = quantize(I_256, 32)
plt.subplot(2,4,4), plt.imshow(I_32), plt.title('32 grey levels'), plt.axis('off')

I_16 = quantize(I_256, 16)
plt.subplot(2,4,5), plt.imshow(I_16), plt.title('16 grey levels'), plt.axis('off')

I_8 = quantize(I_256, 8)
plt.subplot(2,4,6), plt.imshow(I_8), plt.title('8 grey levels'), plt.axis('off')

I_4 = quantize(I_256, 4)
plt.subplot(2,4,7), plt.imshow(I_4), plt.title('4 grey levels'), plt.axis('off')

I_2 = quantize(I_256, 2)
plt.subplot(2,4,8), plt.imshow(I_2), plt.title('2 grey levels'), plt.axis('off')


plt.show()
