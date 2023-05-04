# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:03:51 2020

@author: capliera
"""

import numpy as np
import skimage.io as skio
import skimage.data as skd
import skimage.color as skc
import matplotlib.pyplot as plt
import skimage.filters as skf
import skimage.util as sku
import skimage.metrics as skme
import skimage.morphology as skmo
from skimage.filters.rank import mean_bilateral
from os.path import join
from skimage.morphology import square



#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

house_img=skio.imread("Images/House.jpg")
train_img = skio.imread("Images/Train.jpg")
fig1=plt.figure(1)
plt.subplot(3,2,1), skio.imshow(house_img), plt.title("House"), plt.axis('off')
plt.subplot(3,2,2), skio.imshow(train_img), plt.title("Train"), plt.axis('off')
print("Loaded image has dimensions:", house_img.shape)
print("Loaded values are of type:", house_img.dtype)

# Appropriate filtering with different neighborhood sizes in order to remove 
# salt and pepper noise

# Gaussian filtering
gaussian_house = house_img.copy()
gaussian_house = skf.gaussian(gaussian_house, sigma=1)
gaussian_train = train_img.copy()
gaussian_train = skf.gaussian(gaussian_train, sigma=1)

# Median filtering
median_house = house_img.copy()
median_house = skf.median(median_house)
median_train = train_img.copy()
median_train = skf.median(median_train)

# Image display
plt.subplot(3,2,3), skio.imshow(gaussian_house), plt.title("House with gaussian filter"), plt.axis('off')
plt.subplot(3,2,4), skio.imshow(gaussian_train), plt.title("Train with gaussian filter"), plt.axis('off')
plt.subplot(3,2,5), skio.imshow(median_house), plt.title("House with median filter"), plt.axis('off')
plt.subplot(3,2,6), skio.imshow(median_train), plt.title("Train with median filter"), plt.axis('off')

# Adaptive filtering
def adaptive_filtering(img, tho):
    f_img = img.copy()
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            cp_ip_sum, cp_sum  = sum_neighborhood(img, i, j, tho)
            f_img[i, j] =  cp_ip_sum/cp_sum

    return f_img

def sum_neighborhood(img, i, j, tho):
    cp_ip_sum = 0
    cp_sum = 0
    for k in range(i-1, i+1):
        for l in range(j-1, j+1):
            if abs(img[i,j]-img[k,l]) < tho:
                cp = 1
            else:
                cp = 0
            cp_ip_sum += cp*img[k, l]
            cp_sum += cp

    return cp_ip_sum, cp_sum

# Adaptive filtering with tho = 10
adaptive_house = house_img.copy()
adaptive_house = adaptive_filtering(adaptive_house, 10)
fig2=plt.figure(2)
plt.subplot(1,2,1), skio.imshow(adaptive_house), plt.title("House with adaptive filter"), plt.axis('off')
plt.subplot(1,2,2), skio.imshow(house_img), plt.title("House"), plt.axis('off')


# Image loading and display
image = skd.coffee()
image=skc.rgb2gray(image)
fig3=plt.figure(3)
plt.subplot(2,2,1), skio.imshow(image), plt.title("Coffee"), plt.axis('off')
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)

# Add to the image a Gaussian noise of variance 0.01 by using the random.noise 
# function coming from skimage.util and compute the psnr between the original
# and the noisy images by using the peak_signal_noise_ratio funtion from
# skimage.metrics

#noisy_img = image.copy()
#noisy_img = sku.random_noise(image, mode='gaussian', mean=0, var=0.01)
#plt.subplot(2,2,2), skio.imshow(noisy_img), plt.title("Noisy image"), plt.axis('off')
#print("Loaded image has dimensions:", image.shape)
#print("Loaded values are of type:", image.dtype)
#
#denoised_img1 = image.copy()
#denoised_img1= adaptive_filtering(denoised_img1, 10)
#plt.subplot(2,2,3), skio.imshow(denoised_img1), plt.title("Denoised image with tho=10"), plt.axis('off')
#denoised_img2 = image.copy()
#denoised_img2= adaptive_filtering(denoised_img2, 50)
#plt.subplot(2,2,4), skio.imshow(denoised_img2), plt.title("Denoised image with tho=50"), plt.axis('off')

#Adaptive filtering - Determining the best tau value
# Implement the adaptive filtering for different tau values
# Determine the best tau value by computing the psnr between the original
# and the adaptive filtered images - Display the filtered image corresponding
# to the best tau value

pnsr = []
tho_list = []
denoised_img = image.copy()
for tho in range(10, 250, 10):
    tho_list.append(tho)
    denoised_img = adaptive_filtering(denoised_img, tho)
    pnsr.append(skme.peak_signal_noise_ratio(image, denoised_img))

fig4=plt.figure(4)
plt.plot(tho_list, pnsr), plt.title("PSNR en fonction de tho"), plt.xlabel("tho"), plt.ylabel("PSNR")
plt.show()

# Bilateral filtering - Use the mean_bilateral function in order to filter
# the noisy image and compute the psnr between the origainal and the bilateral
# filtered images

# TO BE COMPLETED







