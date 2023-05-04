# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:17:39 2020

@author: capliera
"""

import skimage.io as skio
from os.path import join
from skimage.filters import threshold_otsu
import skimage as sk
import matplotlib.pyplot as plt
from skimage import color
import skimage.exposure as skiexp 
from skimage.transform import resize
import numpy as np



img1 = skio.imread("Images/book_page_1.jpg")
img2 = skio.imread("Images/book_page_2.jpg")

img1 = color.rgb2gray(img1)
img2 = color.rgb2gray(img2)

img1 = sk.img_as_float(img1)
img2 = sk.img_as_float(img2)


print("Loaded image has dimensions:", img1.shape)

# global thresholding

hist1, hist_centers1 = skiexp.histogram(img1)
hist2, hist_centers2 = skiexp.histogram(img2)

fig1=plt.figure(1)
plt.subplot(2,2,1), plt.plot(hist_centers1, hist1, lw=2), plt.title('Image 1 histogram')
plt.subplot(2,2,2), skio.imshow(img1), plt.title('Image 1 ')
plt.subplot(2,2,3), plt.plot(hist_centers2, hist2, lw=2), plt.title('Image 2 histogram')
plt.subplot(2,2,4), skio.imshow(img2), plt.title('Image 2 ')

t1 = threshold_otsu(img1)
t2 = threshold_otsu(img2)

print("Threshold value for image 1:", t1) # float because image is float
print("Threshold value for image 2:", t2)

# binarization
bin1 = img1 > t1
bin2 = img2 > t2

bin_hist1, bin_hist_centers1 = skiexp.histogram(bin1)
bin_hist2, bin_hist_centers2 = skiexp.histogram(bin2)

fig2=plt.figure(2)
plt.subplot(2,2,1), plt.plot(bin_hist_centers1, bin_hist1, lw=2), plt.title('Image 1 histogram')
plt.subplot(2,2,2), skio.imshow(bin1), plt.title('Image 1 ')
plt.subplot(2,2,3), plt.plot(bin_hist_centers2, bin_hist2, lw=2), plt.title('Image 2 histogram')
plt.subplot(2,2,4), skio.imshow(bin2), plt.title('Image 2 ')


# Binarization improvment: different threshold for each region
#tgt_img = skio.imread("Target_images/road.jpg")
#tgt_img = resize(tgt_img, (820, 1280)) 
#Split the target image in small blocks and compute the mean color of each channel over each block.
#Generate a new image in which the color of all the pixels in a given block are replaced by the mean color value.

def split_grey_image(tgt_img, resolution):
    size = tgt_img.shape # 832x1280x3
    blk_size = (size[0]//resolution, size[1]//resolution)
    new_img_size = (resolution*blk_size[0], resolution*blk_size[1])
    tgt_img = resize(tgt_img, new_img_size)
    print( "New image size:", tgt_img.shape)
    print( "Block size:", blk_size)
    new_img = np.zeros(new_img_size)
    blk_img = np.zeros((blk_size[0], blk_size[1]))
    for i in range(0, new_img_size[0], blk_size[0]):
        for j in range(0, new_img_size[1], blk_size[1]):
            blk_img = tgt_img[i:i+blk_size[0], j:j+blk_size[1]]
            t_blk = threshold_otsu(blk_img)  # calcul otsu threshold
            bin_blk = blk_img > t_blk # seuillage du bloc
            new_img[i:i+blk_size[0], j:j+blk_size[1]] = bin_blk

    return new_img

resolution = 8
new_bin1 = split_grey_image(img1, resolution)
h, hc = skiexp.histogram(new_bin1) 

fig3 = plt.figure(3)
plt.subplot(1,3,1), skio.imshow(new_bin1), plt.title('Image 1 with block thresold')
plt.subplot(1, 3, 3), plt.plot(hc, h, lw=2), plt.title('Image 1 histogram')
plt.show()