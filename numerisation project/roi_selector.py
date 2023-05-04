# -*- coding: utf-8 -*-
"""
Created on Sun May 30 01:27:18 2021

@author: flore
"""
from PIL import Image
import numpy as np
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pandas import read_csv
import validation_functions as vf
from os.path import split


class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        # GRANDE FENETRE
        self.initUI()

    def initUI(self):

        # CALCULATING
        if (len(sys.argv) <= 1):
            raise ValueError(
                "Veuillez rentrer le chemin contenant les images et le nom du fichier csv")
        else:
            self.folder_path = sys.argv[1]
            self.csv_filename = sys.argv[2]
            self.filenames = vf.get_folder_filenames(self.folder_path)
        self.df = read_csv(self.csv_filename, names=[
            'mean_lum', 'std_lum', 'std_grad_magn', 'R'], index_col=0)
        self.mean_lum = 0
        self.std_lum = 0
        self.std_grad = 0
        self.mean_ratio = 0
        self.area_ratio = 0
        self.quad_coords = np.zeros((4, 2))

        # WINDOW TITLE AND LOGO
        self.setWindowTitle("Super 8 Test")
        self.window_width = QDesktopWidget().screenGeometry().width()
        self.window_height = QDesktopWidget().screenGeometry().height()

        # MAIN WIDGET
        window_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(window_layout)

        # HELP BUTTON
        help_button = QPushButton("Aide")
        help_button.setMaximumWidth(100)
        window_layout.addWidget(help_button, alignment=Qt.AlignRight)

        # MAIN UNDER WIDGETS
        parametre_box = QGroupBox("Lancement des tests")
        parametre_box.setMaximumHeight(200)
        window_layout.addWidget(parametre_box)

        images_box = QGroupBox("Image")
        self.max_img_w = images_box.frameGeometry().width()
        self.max_img_h = images_box.frameGeometry().height()
        window_layout.addWidget(images_box)

    # FRAME INTERACTION
        # CHOIX DE L'IMAGE
        choose_us_image = QLabel("Choisir image")
        choose_us_image.setMaximumWidth(200)

        browse_us_image = QPushButton("Parcourir")
        browse_us_image.setMaximumWidth(200)
        browse_us_image.clicked.connect(self.getUsFile)

        # CALCUL DU RATIO
        ratio_label = QLabel("Calcul du ratio image utile")
        ratio_label.setMaximumWidth(200)

        select_roi = QPushButton("Sélectionner cadre utile")
        select_roi.setMaximumWidth(200)

        reset_points = QPushButton("Réinitialiser points")
        reset_points.setMaximumWidth(200)
        reset_points.clicked.connect(self.resetPoints)

        calculate_ratio = QPushButton("Calculer ratio")
        calculate_ratio.setMaximumWidth(200)
        calculate_ratio.clicked.connect(self.calcAreaRatio)

        self.disp_ratio = QLabel()
        self.disp_ratio.setMaximumWidth(100)
        self.disp_ratio.setText("Ratio : " + str(self.area_ratio))

        save_ratio = QPushButton("Sauvegarder ratio")
        save_ratio.setMaximumWidth(200)
        save_ratio.clicked.connect(self.saveRatio)

        # TESTS QUALITE D'IMAGES
        quality_label = QLabel("Tests qualité d'image")
        quality_label.setMaximumWidth(200)

        run_test = QPushButton("Lancer tests")
        run_test.setMaximumWidth(200)
        run_test.clicked.connect(self.run_quality_test)

        self.disp_mean_lum = QLabel()
        self.disp_mean_lum.setMaximumWidth(200)
        self.disp_mean_lum.setText(
            "Luminosité moyenne : " + str(self.mean_lum))

        self.disp_std_lum = QLabel()
        self.disp_std_lum.setMaximumWidth(200)
        self.disp_std_lum.setText(
            "Variance de luminosité : " + str(self.std_lum))

        self.disp_std_grad = QLabel()
        self.disp_std_grad.setMaximumWidth(200)
        self.disp_std_grad.setText(
            "Variance du gradient : " + str(self.std_grad))

        self.disp_mean_ratio = QLabel()
        self.disp_mean_ratio.setMaximumWidth(200)
        self.disp_mean_ratio.setText(
            "Ratio image utile moyen : " + str(self.mean_ratio))

        # Ajout des boutons du frame parametres
        parametres_layout = QGridLayout(parametre_box)

        parametres_layout.addWidget(choose_us_image, 1, 0)
        parametres_layout.addWidget(browse_us_image, 1, 1)

        parametres_layout.addWidget(ratio_label, 2, 0)
        parametres_layout.addWidget(select_roi, 2, 1)
        parametres_layout.addWidget(reset_points, 2, 2)
        parametres_layout.addWidget(calculate_ratio, 2, 3)
        parametres_layout.addWidget(self.disp_ratio, 2, 4)
        parametres_layout.addWidget(save_ratio, 2, 5)

        parametres_layout.addWidget(quality_label, 3, 0)
        parametres_layout.addWidget(run_test, 3, 1)
        parametres_layout.addWidget(self.disp_mean_lum, 3, 2)
        parametres_layout.addWidget(self.disp_std_lum, 3, 3)
        parametres_layout.addWidget(self.disp_std_grad, 3, 4)
        parametres_layout.addWidget(self.disp_mean_ratio, 3, 5)

    #  FRAME AFFICHAGE D'IMAGE
        # US IMAGE FRAME
        us_img_frame = QFrame()
        self.configureImageFrame(us_img_frame)

        us_img_label = QLabel("Image", us_img_frame)
        us_img_label.setAlignment(Qt.AlignCenter)
        self.us_img_view = QLabel(us_img_frame)
        self.us_img_view.setAlignment(Qt.AlignCenter)

        self.configureImageLayout(us_img_frame, us_img_label, self.us_img_view)

        # ALL IMAGES LAYOUT
        image_layout = QGridLayout(images_box)
        image_layout.addWidget(us_img_frame, 0, 0)

    # DISPLAY WIDGETS
        self.setCentralWidget(widget)

    # PAINTING
        self.us_img_array = np.zeros(
            (self.max_img_h, self.max_img_w, 3), dtype=np.uint8)
        self.mPixmap = QPixmap()

        self.center_point = QPoint()
        self.center_point.setX(
            round(images_box.frameGeometry().width()))
        self.center_point.setY(
            round(images_box.frameGeometry().height()/2 + 100))

        self.drawing_point = QPoint()

    def configureImageFrame(self, frame):
        frame.setFrameStyle(QFrame.Box)  # frame us_img
        frame.setLineWidth(1)
        frame.setFrameShadow(QFrame.Sunken)

    def configureImageLayout(self, parent, text_label, image_view):
        layout = QGridLayout(parent)
        layout.addWidget(text_label, 0, 1, 1, 1, alignment=Qt.AlignTop)
        layout.addWidget(image_view, 1, 0, 3, 3)
        return layout

    def getUsFile(self):
        self.us_filename = QFileDialog.getOpenFileName(self, "Open image")[0]
        if self.us_filename != '':
            self.us_img_array = prepare_image(
                self.us_filename, self.max_img_h, self.max_img_h)

            self.dispUsImage(self.us_img_array)

    def dispUsImage(self, img_array_to_disp):
        h, w, ch = img_array_to_disp.shape
        bytesPerLine = ch * w
        self.q_us_img = QImage(img_array_to_disp, w, h,
                               bytesPerLine, QImage.Format_RGB888)
        self.mPixmap = QPixmap.fromImage(self.q_us_img)
        self.center_point.setX(
            round(self.window_width/2 - self.mPixmap.width()/2))
        self.center_point.setY(
            round(self.window_height/2 - self.mPixmap.height()/2))
        self.painter.drawPixmap(self.center_point, self.mPixmap)

    def resetPoints(self):
        self.quad_coords = np.zeros((4, 2))

    def addCoord(self, drawing_point):
        # regarde quelle case pas encore occupée
        point_added = False
        for i in range(4):
            if self.quad_coords[i, 0] == 0 and self.quad_coords[i, 1] == 0:
                self.quad_coords[i, 0] = drawing_point.x() - \
                    self.center_point.x()
                self.quad_coords[i, 1] = drawing_point.y() - \
                    self.center_point.y()
                point_added = True
                break  # on sort de la boucle si on a rempli une case du tableau

        return point_added

    def calcAreaRef(self):
        return self.mPixmap.height() * self.mPixmap.width()

    def calcAreaRatio(self):
        self.area_ratio = area(self.quad_coords)/self.calcAreaRef()
        self.disp_ratio.setText("Ratio : " + str(self.area_ratio))

    def saveRatio(self):

        self.df.R[self.us_filename.split("/")[-1]] = self.area_ratio
        self.df.to_csv(self.csv_filename, header=False,
                       index_label=self.filenames)

    def run_quality_test(self):
        ''' Lance tests de luminosité et de flou sur les images'''
        # Calculs sur chaque image
        self.df = vf.calc_mean_std_luminosity(
            self.folder_path, self.filenames, self.df)
        self.df = vf.calc_std_gradient(
            self.folder_path, self.filenames, self.df)

        self.df.to_csv(self.csv_filename, header=False,
                       index_label=self.filenames)  # sauvegarder le dataframe

        # Calcul des moyennes
        self.mean_lum = self.df['mean_lum'].mean()
        self.std_lum = self.df['std_lum'].mean()
        self.std_grad = self.df['std_grad_magn'].mean()
        self.mean_ratio = self.df['R'].mean()

        # Afficher résultats
        self.disp_mean_lum.setText(
            "Luminosité moyenne : " + str(self.mean_lum))
        self.disp_std_lum.setText(
            "Variance de luminosité : " + str(self.std_lum))
        self.disp_std_grad.setText(
            "Variance du gradient : " + str(self.std_grad))
        self.disp_mean_ratio.setText(
            "Ratio image utile moyen : " + str(self.mean_ratio))

    def paintEvent(self, event):
        super().paintEvent(event)
        self.painter = QPainter(self)
        self.dispUsImage(self.us_img_array)
        self.painter.setPen(QPen(QColor(141, 205, 204), 5,
                            Qt.SolidLine))  # peppermint
        self.painter.setRenderHint(QPainter.Antialiasing)  # smooth edges
        if self.drawing_point.x() != 0 and self.drawing_point.y() != 0:
            self.painter.drawPoint(
                self.drawing_point.x(), self.drawing_point.y())
        self.painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing_point = QPoint(event.pos())
            if self.addCoord(self.drawing_point) == False:  # on arrête
                print("ratio area = ", self.calcAreaRatio())
            self.update()
        super().mousePressEvent(event)  # on fait un override de la méthode mousePressEvent


def save_image(seg_img_array, seg_filename):
    seg_img = Image.fromarray(seg_img_array)
    seg_img = seg_img.convert("RGB")
    seg_img.save(seg_filename)


def prepare_image(img_path, max_w, max_h):
    '''Prepares the image to be displayed on a window. Reduces the size array to the nearest width or height coeficient if necessary.
    Parameters
    ----------
    img_path: str
        Path leading to the image to display
    max_w: int
        Maximum width wanted
    max_h: int
        Maximum height wanted

    Returns
    ----------
    img_array: np.array
        3D Array of the image to display downsampled to the size of the window '''

    # Get the image and its size
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    img_h = img_array.shape[0]
    img_w = img_array.shape[1]

    if img_w > max_w or img_h > max_h:
        # set the base width or height
        if max_w/img_w < max_h/img_h:  # si le coef est + petit alors on prend celui là comme base
            final_width = max_w
            final_height = round(img_h*(final_width/img_w))
        else:
            final_height = max_h
            final_width = round(img_w*(final_height/img_h))
        # resize
        img = img.resize((final_width, final_height))

    return np.array(img)


def area(quad_coords):
    p1 = quad_coords[0]
    p2 = quad_coords[1]
    p3 = quad_coords[2]
    p4 = quad_coords[3]
    print(0.5*np.abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p4[1] + p4[0]
          * p1[1] - p2[0]*p1[1] - p3[0]*p2[1] - p4[0]*p3[1] - p1[0]*p4[1]))
    return 0.5 * np.abs(p1[0]*p2[1] + p2[0]*p3[1] + p3[0]*p4[1] + p4[0]*p1[1] - p2[0]*p1[1] - p3[0]*p2[1] - p4[0]*p3[1] - p1[0]*p4[1])


def main():

    app = QApplication(sys.argv)
    ex = Example()
    ex.showMaximized()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
