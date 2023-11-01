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
        self.ui.btn2_Bilateral.clicked.connect(self.BilateralFilter)
        self.ui.btn2_Median.clicked.connect(self.MedianFilter)
        self.ui.btn3_SobelX.clicked.connect(self.SobelX)
        self.ui.btn3_SobelY.clicked.connect(self.SobelY)
        self.ui.btn3_CombAndThres.clicked.connect(self.CombAndThres)


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

    def BilateralFilter(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        cv2.imshow('bilateral filter', image)

        cv2.createTrackbar('m', 'bilateral filter', 1, 5, self.update_bilateral_filter)
        cv2.setTrackbarMin('m', 'bilateral filter', 1)  # Set the minimum value
        cv2.setTrackbarMax('m', 'bilateral filter', 5)  # Set the maximum value
        cv2.setTrackbarPos('m', 'bilateral filter', 1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def update_bilateral_filter(self, value):
        self.m = value
        if self.Image is not None:
            image = cv2.imread(self.Image)
            window_radius = 2 * self.m + 1
            filter_image = cv2.bilateralFilter(image, window_radius,90,90)
            cv2.imshow('bilateral filter', filter_image)

    def MedianFilter(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        cv2.imshow('median filter', image)

        cv2.createTrackbar('m', 'median filter', 1, 5, self.update_median_filter)
        cv2.setTrackbarMin('m', 'median filter', 1)  # Set the minimum value
        cv2.setTrackbarMax('m', 'median filter', 5)  # Set the maximum value
        cv2.setTrackbarPos('m', 'median filter', 1)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def update_median_filter(self,value):
        self.m = value
        if self.Image is not None:
            image = cv2.imread(self.Image)
            window_radius = 2 * self.m + 1
            filter_image = cv2.medianBlur(image, window_radius)
            cv2.imshow('median filter', filter_image)

# 3. Edge Detection

    def SobelX(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        # Sobel x operator for edge detection
        sobel_x_filter = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Create an empty output image
        sobel_x_image = np.zeros_like(smooth_image)
        # Iterate through the image and apply the Sobel x operator
        for y in range(1, smooth_image.shape[0] - 1):
            for x in range(1, smooth_image.shape[1] - 1):
                sobel_x_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * sobel_x_filter)
                sobel_x_image[y, x] = np.abs(sobel_x_pixel) #the pixel value at position (y,x)

        # normalize between [0,255]
        min_value = np.min(sobel_x_image)
        max_value = np.max(sobel_x_image)
        sobel_x_image = np.uint8(255 * (sobel_x_image / (max_value - min_value)))

        # Display the Sobel x image
        cv2.imshow('Sobel X Image', sobel_x_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def SobelY(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        # Sobel y operator for edge detection
        sobel_y_filter = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Create an empty output image
        sobel_y_image = np.zeros_like(smooth_image)
        # Iterate through the image and apply the Sobel y operator
        for y in range(1, smooth_image.shape[0] - 1):
            for x in range(1, smooth_image.shape[1] - 1):
                sobel_y_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * sobel_y_filter)
                sobel_y_image[y, x] = np.abs(sobel_y_pixel)  # the pixel value at position (y,x)

        # normalize between [0,255]
        min_value = np.min(sobel_y_image)
        max_value = np.max(sobel_y_image)
        sobel_y_image = np.uint8(255 * (sobel_y_image / (max_value - min_value)))

        # Display the Sobel x image
        cv2.imshow('Sobel Y Image', sobel_y_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def CombAndThres(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        sobel_x_filter = np.array([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]])
        sobel_y_filter = np.array([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]])
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Create an empty output image
        sobel_image = np.zeros_like(smooth_image)
        # Iterate through the image and apply the Sobel y operator
        for y in range(1, smooth_image.shape[0] - 1):
            for x in range(1, smooth_image.shape[1] - 1):
                sobel_x_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * sobel_x_filter)
                sobel_y_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * sobel_y_filter)
                sobel_image[y, x] = np.sqrt(np.square(sobel_x_pixel)+np.square(sobel_y_pixel))  # the pixel value at position (y,x)

        # normalize between [0,255]
        min_value = np.min(sobel_image)
        max_value = np.max(sobel_image)
        sobel_image = np.uint8(255 * (sobel_image / (max_value - min_value)))

        # Display the Combination image
        cv2.imshow('Combination of Sobel x and Sobel y', sobel_image)

        threshold = 128
        ret, thres_image = cv2.threshold(sobel_image, threshold, 255, cv2.THRESH_BINARY)    # 0, if lower than 128
        cv2.imshow('Threshold result', thres_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def GradientAngle(self):
        if self.Image is None:
            return








