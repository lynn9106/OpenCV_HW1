from MainUI import Ui_mainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import os.path
import cv2
import numpy as np

class MainWindow_controller(QtWidgets.QMainWindow):

    Image = None

    def __init__(self):
        super().__init__()
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.btn_LoadImage.clicked.connect(self.openfile1)
        self.ui.btn1_CSeperation.clicked.connect(self.ColorSeperate)
        self.ui.btn1_CTransformation.clicked.connect(self.ColorTransformation)
        self.ui.btn1_CExtraction.clicked.connect(self.ColorExtraction)
        self.ui.btn2_Gaussian.clicked.connect(self.GaussianBlur)


    def openfile1(self):
        filepath, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")                 # start path
        self.Image = filepath
        filename = os.path.basename(filepath)
        self.ui.label_ImageName.setText(filename)

# 1. Image Processing
    def ColorSeperate(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        zeros = np.zeros(image.shape[:2], dtype="uint8")
        # Split the BGR channels
        blue_channel, green_channel, red_channel = cv2.split(image)

        B_image = cv2.merge([blue_channel, zeros, zeros])
        G_image = cv2.merge([zeros, green_channel, zeros])
        R_image = cv2.merge([zeros, zeros, red_channel])

        cv2.imshow('B Channel', B_image)
        cv2.imshow('G Channel', G_image)
        cv2.imshow('R Channel', R_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ColorTransformation(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('I1(OpenCV function)', gray_image)

        blue_channel, green_channel, red_channel = cv2.split(image)
        Average_image = (blue_channel+green_channel+red_channel)/3
        cv2.imshow('I2(Averaged Weighted)', Average_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ColorExtraction(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    # transform BGR to HSV

        yellow_green_Mask = cv2.inRange(hsv_image,(15,50,25) ,(80,255,255))     # Extract Yellow-Green mask
        cv2.imshow('I1(mask)', yellow_green_Mask)

        yellow_green_BGR = cv2.cvtColor(yellow_green_Mask, cv2.COLOR_GRAY2BGR)  # Convert the Yellow-Green mask to BGR format
        Remove_YGMask = cv2.bitwise_not(yellow_green_BGR, image, yellow_green_Mask)     #Remove Yellow and Green color from the original image using the mask
        cv2.imshow('I2(Image without yellow and green)', Remove_YGMask)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 2. Image Smoothing

    m = 1
    def GaussianBlur(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        cv2.imshow('gaussian blur', image)

        cv2.createTrackbar('m', 'gaussian blur', 1, 5, self.update_gaussian_blur)
        cv2.setTrackbarMin('m', 'gaussian blur', 1)  # Set the minimum value
        cv2.setTrackbarMax('m', 'gaussian blur', 5)  # Set the maximum value
        cv2.setTrackbarPos('m', 'gaussian blur', 1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def update_gaussian_blur(self, value):
        self.m = value
        if self.Image is not None:
            image = cv2.imread(self.Image)
            window_radius = 2 * self.m + 1
            blur_image = cv2.GaussianBlur(image, (window_radius, window_radius), 0)
            cv2.imshow('gaussian blur', blur_image)

# WHY INITIAL IS 0 ??????????



'''
    def window_radius(image):
        global gaussian_m
        image = cv2.imread(image)
        window_radius = 2 * gaussian_m + 1
        blur_image = cv2.GaussianBlur(image, (window_radius,window_radius), 0)
        cv2.imshow('gaussian blur', blur_image)

'''









