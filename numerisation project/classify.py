'''Program which separates the images into two folders: good and bad. 
The images are classified as good if we don't see the edge between two images of the film in the center of the acquired image.
Otherwise, the image is classified as bad.    
 '''

import validation_functions as vf
import sys
import skimage.io as skio
import cProfile
import re


def main():
    # extraire folder_path et csv_filename de la ligne de commande
    if len(sys.argv) == 3:
        folder_path = sys.argv[1]
        new_folder_path = sys.argv[2]
    else:
        print("Error: wrong number of arguments")
        sys.exit(1)

    filenames = vf.get_folder_filenames(folder_path)
    img_height = skio.imread(folder_path + "/" + filenames[0]).shape[0]
    print(img_height)
    ref_img = skio.imread("ref_img.png", as_gray=True)
    all_y = vf.compute_all_y_distance(folder_path, filenames, ref_img)
    vf.plot_histogramme(all_y)

    good_img_percentage = vf.estimate_good_img_percentage(all_y)
    print("Percentage of good images: ", good_img_percentage)
    good_img, bad_img = vf.classify_img(all_y, filenames, img_height)
    print("Number of good images: ", len(good_img))
    print("Number of bad images: ", len(bad_img))
    vf.create_good_img_folder(good_img, folder_path, new_folder_path)


if __name__ == '__main__':
    # cProfile.run('main()')
    main()
