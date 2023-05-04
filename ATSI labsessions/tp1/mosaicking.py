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
from skimage.transform import resize
import math

import skimage as sk
import skimage.data as skd
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join

tgt_img = skio.imread("Target_images/road.jpg")
tgt_img = resize(tgt_img, (820, 1280))
# Split the target image in small blocks and compute the mean color of each channel over each block.
# Generate a new image in which the color of all the pixels in a given block are replaced by the mean color value.


def split_image(tgt_img, blk_size=(20, 32, 3)):
    size = tgt_img.shape  # 832x1280x3
    final_img_size = (blk_size[0]*blk_size[1], blk_size[0], blk_size[1], 3)
    #all_blk_img = np.zeros(final_img_size)
    new_img = np.zeros(size)
    blk_img = np.zeros((blk_size[0], blk_size[1], 3))
    for i in range(0, size[0], blk_size[0]):
        for j in range(0, size[1], blk_size[1]):
            blk_img = tgt_img[i:i+blk_size[0], j:j+blk_size[1], :]
            mean_color = (blk_img[:, :, 0].mean(
            ), blk_img[:, :, 1].mean(), blk_img[:, :, 2].mean())
            new_img[i:i+blk_size[0], j:j+blk_size[1], :] = mean_color
            #np.append(all_blk_img, np.reshape(blk_img, (1, blk_size[0], blk_size[1], 3)), axis=0)
    return new_img


blk_img = split_image(tgt_img)

fig2 = plt.figure(2)
plt.subplot(1, 3, 1), plt.imshow(tgt_img), plt.title(
    "original image"), plt.axis("off")
plt.subplot(1, 3, 2), plt.imshow(blk_img), plt.title(
    "block image"), plt.axis("off")

# For each patch in the dataset, compute the mean color on each R, G and B channel.
mean_colors = []
# ouvrir toutes les images du dataset
input_img = [f for f in listdir("input") if isfile(join("input", f))]
# pour chaque image, calculer la moyenne de chaque canal et stocker les moyennes dans une liste
for img_f in input_img:
    img = skio.imread(join("input", img_f))
    img = img_as_float(img)
    mean_color = np.mean(img, axis=(0, 1))
    mean_colors.append(mean_color)

# Select in the patch dataset, the patch that is the most similar to the target block to be replaced.
# The Euclidean distance will be used in order to compute the color similarity between each patch
# and a given block. Generate a second image in which each block has been replaced by the most
# similar patch that has been resized to fit the block size.
# pour chaque bloc de l'image cible, calculer la distance euclidienne entre la couleur moyenne du bloc et celle du patch


def generate_img(blk_img, mean_colors, blk_size=(20, 32, 3)):

    new_img = np.zeros(blk_img.shape)
    for i in range(0, blk_img.shape[0], blk_size[0]):
        for j in range(0, blk_img.shape[1], blk_size[1]):
            blk_img_color = blk_img[i, j, :]
            # print(blk_img_color)
            dist = [np.sqrt(sum((blk_img_color - mean_color)**2))
                    for mean_color in mean_colors]
            min_dist = min(dist)
            min_k = dist.index(min_dist)
            new_img[i:i+blk_size[0], j:j+blk_size[1], :] = img_as_float(
                resize(skio.imread(join("input", input_img[min_k])), (20, 32, 3)))
    return new_img


new_img = generate_img(blk_img, mean_colors)
plt.subplot(1, 3, 3), plt.imshow(
    new_img), plt.title("new image"), plt.axis("off")
plt.show()
