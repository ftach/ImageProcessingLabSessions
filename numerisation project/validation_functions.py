''' Functions that enables to compute all the metrics on the image dataset.
'''

import cv2
import numpy as np
from os import listdir, path, makedirs
import skimage.filters as skf
import skimage.io as skio
import scipy.signal as sig
import matplotlib.pyplot as plt


def get_folder_filenames(folder_path):
    ''' Puts all the image filename in a list to open them afterwards.
    Parameters
    - folder_path: str
        pathway of the folder containing all the images
    Returns
    - all_filenames: list of str
        List containing all the filename of the image contained in the folder
    '''

    all_filenames = [f for f in listdir(
        folder_path) if path.isfile(path.join(folder_path, f))]

    return all_filenames


def calc_mean_std_luminosity(folder_path, filenames, df):
    ''' Computes and save in a csv file the mean and standard deviation luminosity of the whole image dataset. 
    Parameters
    - folder_path: str 
        Pathway of the folder containing the images 
    - filenames: list of str
        Filenames of the different images of the dataset
    Returns
        None 
    '''

    for img_path in filenames:
        full_img_path = folder_path + "/" + img_path
        img = cv2.imread(full_img_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_v = img_hsv[:, :, 2]
        img_mean_luminosity = np.mean(img_v)
        img_std_luminosity = np.std(img_v)

        # modifier case dans tableau pandas
        df.mean_lum[img_path] = img_mean_luminosity
        df.std_lum[img_path] = img_std_luminosity

    # df.to_csv(csv_filename, mode='a')  # enregistrer nouveau csv

    return df


def std_gradient(img):
    ''' Computes the standard deviation of the magnitude gradient of a given image. 
    Parameters
    - img: ?
        Image array loaded with cv2 or skio 
    Return 
    - img_std_mag: float 
        Standard deviation of the image gradient magnitude 
    '''

    gx = skf.sobel_h(img)
    gy = skf.sobel_v(img)
    mag = np.sqrt(gx**2 + gy**2)

    return np.var(mag)


def calc_std_gradient(folder_path, filenames, df):
    ''' Computes and save in a csv_file the standard deviation gradient magnitude of all the image contained in the dataset. 
    Parameters
    - folder_path: str 
        Pathway of the folder containing the images 
    - filenames: list of str
        Filenames of the different images of the dataset
    Returns 
    None 
    '''
    for img_path in filenames:
        full_img_path = folder_path + "/" + img_path
        img = cv2.imread(full_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        df.std_grad_magn[img_path] = std_gradient(img)

    return df


def get_y_distance(img, ref_img):
    ''' Computes the distance between the reference image and the image of the dataset.
    Parameters
    - img: array
        Image of the dataset
    - ref_img: array
        Reference image of the dataset
    Returns 
    - y: int
        Distance between the reference image and the image of the dataset
    '''
    # Centrage des images pour éviter les problèmes de biais
    img = img - np.mean(img)
    ref_img_centred = ref_img - np.mean(ref_img)

    # Calcul de la corrélation
    corr = sig.correlate(img, ref_img_centred, mode='same')

    y = np.where(np.max(corr) == corr)[0][0]  # Récupération de la distance

    return y


def compute_all_y_distance(folder_path, all_filenames, ref_img):
    ''' Computes all the y position of the black edge in the images of the dataset.
    Parameters
    - folder_path: str
        Pathway of the folder containing the images
    - all_filenames: list of str
        Filenames of the different images of the dataset
    - ref_img: array
        Reference image of the dataset  
    Returns 
    - all_y: list of int
        List containing the position in the y axis of the different images of the dataset   
    '''

    all_y = []
    for img_path in all_filenames:
        full_img_path = folder_path + "/" + img_path
        img = skio.imread(full_img_path, as_gray=True)
        # ajouter la position en y du trait noir
        all_y.append(get_y_distance(img, ref_img))

    return all_y


def get_histogramme(vector):
    ''' Computes the histogram of a given vector.
    Parameters
    - vector: list of int
        Vector containing the position in the y axis of the different images of the dataset
    Returns
    - hist: array
        Histogram of the vector
    '''

    hist = np.zeros(max(vector)+1)
    for i in range(len(vector)):
        hist[int(vector[i])] += 1

    return hist


def plot_histogramme(vector):
    ''' Plots the histogram of a given vector.
    Parameters
    - vector: list of int
        Vector containing the position in the y axis of the different images of the dataset
    Returns
    None
    '''

    plt.plot(get_histogramme(vector))
    plt.legend("Histogramme des positions en y du trait noir sur les images")
    plt.xlabel("Position en y du trait noir sur l'image")
    plt.ylabel("Nombre d'images")
    plt.show()


def estimate_good_img_percentage(all_y, img_height=1400):
    ''' Estimates the percentage of good images in the dataset.
    Parameters
    - all_y: list of int
        List containing the position in the y axis of the different images of the dataset
    Returns
    - good: float
        Percentage of good images in the dataset
    '''

    good = 0
    for y in all_y:
        if y < img_height/6 or y > 5*img_height/6:
            good += 1

    return good/len(all_y)*100


def classify_img(all_y, filenames, img_height):
    ''' Classifies the images of the dataset in good and bad images.
    Parameters
    - all_y: list of int
        List containing the position in the y axis of the different images of the dataset
    - filenames: list of str
        Filenames of the different images of the dataset
    Returns
    - good_img: list of str
        List containing the filenames of the good images of the dataset
    - bad_img: list of str
        List containing the filenames of the bad images of the dataset
    '''

    good_img = []
    bad_img = []
    for i in range(len(all_y)):
        if all_y[i] < img_height/6 or all_y[i] > 5*img_height/6:
            good_img.append(filenames[i])
        else:
            bad_img.append(filenames[i])
    return good_img, bad_img


def create_good_img_folder(good_img, folder_path, new_folder_path):
    ''' Creates a folder containing the good images of the dataset.
    Parameters
    - good_img: list of str
        List containing the filenames of the good images of the dataset
    - folder_path: str
        Pathway of the folder containing the images
    - new_folder_path: str
        Pathway of the folder containing the good images of the dataset
    Returns
    None
    '''
    if not path.exists(new_folder_path):
        makedirs(new_folder_path)
    for img_path in good_img:
        full_img_path = folder_path + "/" + img_path
        img = skio.imread(full_img_path)
        skio.imsave(new_folder_path + "/" + img_path, img)

    return path.exists(new_folder_path)
