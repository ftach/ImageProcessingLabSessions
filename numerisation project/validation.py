'''Program which checks the quality of the images acquired with the Raspberry PiCamera. The different paramaters checked are:
    - the mean and variance luminosity
    - the variance of the magnitude gradient
    - the mean ratio useful image area/ image area
    
 '''

import validation_functions as vf
import sys
from pandas import DataFrame, read_csv


def main():
    # extraire folder_path et csv_filename de la ligne de commande
    args = sys.argv[1:]

    if len(args) != 2:
        print("Error: wrong number of arguments")
        sys.exit(1)

    folder_path = args[0]
    csv_filename = args[1]

    # Mettre les valeurs de mean_lum, std_lum, std_grad_magn et R pour chaque photo dans le df
    filenames = vf.get_folder_filenames(folder_path)
    df = DataFrame(columns=['mean_lum', 'std_lum', 'std_grad_magn', 'R'],
                   index=filenames)
    df = vf.calc_mean_std_luminosity(
        folder_path, filenames, df)
    df = vf.calc_std_gradient(folder_path, filenames, df)
    # sauvegarder le dataframe
    df.to_csv(csv_filename, header=False, index_label=filenames)


if __name__ == '__main__':
    main()
