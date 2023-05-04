# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:56:53 2020

@author: capliera
"""

import numpy as np
import skimage.data as skd
import skimage.io as skio
import matplotlib.pyplot as plt


import skimage.exposure as ske


#All figures closing
plt.close('all')

# matplotlib for gray level images display: fix the colormap and 
# the image figure size

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams["figure.figsize"] = (6,4)

# Several test images are available in the scikit-images.data module.
# Those images are entiltled: coins, astronaut (color), brick, camera, cell,
# chekerboard, chelsea (color), clock, coffee (color), colorwheel (color), grass (color),
# gravel (color), horse (black and white), moon, retina (color), rocket (color).

# To download any of those images just use skd.image_name()
# To download any other image, use skio.imread('image_name')


## Grey level image reading and information

I=skd.coins()
Dim = I.shape
nb_pixels=I.size
Max_grey_level=np.max(I)
Min_grey_level=np.min(I)
Mean_grey_level=np.mean(I)

# Image display
fig1=plt.figure(1)                    
skio.imshow(I)
plt.title('Coins')
#plt.show()
#print('Image size: ', Dim) # 303 x 384
#print('Max grey level: ', Max_grey_level) # 252
#print('Min grey level: ', Min_grey_level) # 1
#print('Mean grey level: ', Mean_grey_level) # 96,9

# Color image reading and RGB channel visualization
J=skd.coffee()
# compute the luminance as 0.3*R+0.58*G+0.11*B
K = J.copy()
K[:,:,0] = 0.3*J[:,:,0]
K[:,:,1] = 0.58*J[:,:,1]
K[:,:,2] = 0.11*J[:,:,2]

fig4=plt.figure(4)
plt.subplot(2,1,1), plt.imshow(J), plt.title('Color img')
plt.subplot(2
,1,2), plt.imshow(K), plt.title('L img')
print(K.shape)

#plt.show()

# LUMINANCE EXTRACTION 
# WARNING : instructon A=B does not create a new array, A and B are the same 
# object then modifying A will also modify B => use .copy() function in order
# duplicate an image

# Display the luminance evolution along line 200
luminance = np.sum(K, axis=2)
fig5 = plt.figure(5)
plt.plot(luminance[200,:])
#plt.show()

#Two imshow() functions for image display
# Compute the histogram of image 3 (cf. ske.histogram())
#Display the hitogram ((cf. plt.bar())

image3=skio.imread('Images/CH0SRC.TIF')

# Display image3 by suing first skio.image() and second by usig plt.imshow()
fig6=plt.figure(6) 
plt.subplot(1, 2, 1), plt.imshow(image3), plt.title("Pyplot Image") # darker than the one below, avatn d'afficher l'image elle regarde si toute l'échelle de gris a été utilisée et sinon elle fait une égalisation d'histograamme
plt.subplot(1, 2, 2), skio.imshow(image3), plt.title("Skimage Image") # 
plt.show()

