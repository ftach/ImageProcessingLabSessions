# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:56:10 2020

@author: capliera
"""

import numpy as np
import scipy.fftpack as sc
import scipy.signal as sig
import skimage.io as skio
import skimage.color as skc
import skimage.data as skd
import skimage.filters as skf
import matplotlib.pyplot as plt
from os.path import join
from skimage.util import img_as_float
from matplotlib import cm

plt.rcParams['image.cmap'] = 'gray'
plt.close('all')

# Transfer function of an ideal low pass filter
def filtrePB_Porte(fc,taille):
    s=(taille[0],taille[1])
    H=np.zeros(s)
    [U, V]=np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]]
    H = np.sqrt(U*U+V*V)<fc
    #fc est le rayon de coupure
    return H  

def filtrebutterworth(fc,taille,ordre=1,fig=0):
    s=(taille[0],taille[1])
    H=np.zeros(s)
    [U, V]=np.ogrid[-0.5:0.5:1.0/taille[0],-0.5:0.5:1.0/taille[1]]
    H =1.0/ (1.0+0.414*np.power(np.sqrt(U*U+V*V)/(fc),2*ordre))
    
    U1=np.copy(U[::10,::10])
    V1=np.copy(V[::10])
    H1=np.copy(H[::10])
    if fig>0:
        
        fig = plt.figure(fig)
        ax = fig.gca(projection='3d')
        ax.plot_surface(U1,V1,H1, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_zlim(0, 1.01)

        stringfc=np.str(fc)
        stringordre=np.str(ordre)
        plt.title('Butterworth filter fc='+stringfc+', ordre='+stringordre)

    return H

# Image Fourier Transform
def fourier(image):
    TFI=sc.fft2(image)
    TFI=sc.fftshift(TFI) # mettre 0 au centre de l'image
    spectre=np.abs(TFI**2)
    return TFI,spectre
    
#Image loadind and display
# Plot in the same figure the image and its Fourier transform

image = skd.coffee()
image=skc.rgb2gray(image)
print("Loaded image has dimensions:", image.shape)
print("Loaded values are of type:", image.dtype)



TFI,spectre=fourier(image)

fig1=plt.figure(figsize=(12,8))                    
plt.subplot(2,2,1), plt.imshow(image)
plt.subplot(2,2,2),plt.imshow(np.log10(1+spectre))

def apply_filter(filter, tfi_img):
    """Apply a filter to a Fourier transform of an image
    Parameters:
        filter: the filter to apply
        tfi_img: the Fourier transform of the image
    Returns:
        The filtered Fourier transform and its spectrum
"""
    fimage = np.multiply(tfi_img, filter)
    fspectre = np.abs(fimage**2)
    fimage = sc.fftshift(fimage)
    fimage = sc.ifft2(fimage)
    fimage = np.real(fimage)

    return fimage, fspectre


# Display the ideal low pass filter transfer functions
taille=np.shape(image)
#fig2 = plt.figure(2)
filtre_pb=filtrePB_Porte(0.05,taille)
fimage1,fspectre1 = apply_filter(filtre_pb, TFI)
filtre_bw = filtrebutterworth(0.05,taille,ordre=5,fig=0)
fimage2, fspectre2 = apply_filter(filtre_bw, TFI)
#print("Filtered image shape: ", fimage.shape, " and type: ", fimage.dtype)
fig3=plt.figure(3)    
plt.subplot(2,2,1), plt.imshow(fimage1),plt.title('Low pass filter fc=0.05')
plt.subplot(2,2,2), plt.imshow(np.log10(1+fspectre1)),plt.title('Spectrum after low pass filter fc=0.05')
plt.subplot(2,2,3), plt.imshow(fimage2),plt.title('Butterworth filter fc=0.05, ordre=1')
plt.subplot(2,2,4), plt.imshow(np.log10(1+fspectre2)),plt.title('Spectrum after Butterworth filter fc=0.05, ordre=1')         

plt.show()


# Generate several transfer functions of Buetterworth filters with different 
# parameters. Comment

