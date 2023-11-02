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
        self.ui.btn3_Gradient.clicked.connect(self.GradientAngle)
        self.ui.btn4_Transforms.clicked.connect(self.Transform)


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

    def Sobel_Func(self,filter):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        height, width = smooth_image.shape
        sobel_image = np.zeros((height, width), dtype=np.int32)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                sobel_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * filter)
                sobel_image[y, x] = np.sqrt(np.square(sobel_pixel)) #the pixel value at position (y,x)


        # normalize between [0,255]
        min_value = np.min(sobel_image)
        max_value = np.max(sobel_image)
        sobel_image = np.uint8( sobel_image / (max_value - min_value) * 255)
        return sobel_image

    def Sobel(self,filter):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        height, width = smooth_image.shape
        sobel_image = np.zeros((height, width), dtype=np.int32)


        for y in range(1, height - 1):
            for x in range(1, width - 1):
                sobel_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * filter)
                sobel_image[y, x] = sobel_pixel

        return sobel_image


    # Sobel x filter for edge detection
    sobel_x_filter = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    # Sobel y filter for edge detection
    sobel_y_filter = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    def SobelX(self):
        if self.Image is None:
            return

        sobel_x_image = self.Sobel(self.sobel_x_filter)

        sobel_x_image = np.abs(sobel_x_image)

        min_value = np.min(sobel_x_image)
        max_value = np.max(sobel_x_image)
        sobel_x_image = np.uint8( sobel_x_image / (max_value - min_value) * 255)

        # Display the Sobel x image
        cv2.imshow('Sobel X Image',sobel_x_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def SobelY(self):
        if self.Image is None:
            return

        sobel_y_image = self.Sobel(self.sobel_y_filter)

        sobel_y_image = np.abs(sobel_y_image)

        min_value = np.min(sobel_y_image)
        max_value = np.max(sobel_y_image)
        sobel_y_image = np.uint8( sobel_y_image / (max_value - min_value) * 255)

        # Display the Sobel x image
        cv2.imshow('Sobel Y Image',sobel_y_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def CombAndThres(self):
        if self.Image is None:
            return
        combine_image = self.CombineSobel()

        # Display the Combination image
        cv2.imshow('Combination of Sobel x and Sobel y', combine_image)

        threshold = 128
        ret, thres_image = cv2.threshold(combine_image, threshold, 255, cv2.THRESH_BINARY)    # 0, if lower than 128
        cv2.imshow('Threshold result', thres_image)

        # Wait for a key press and close all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def CombineSobel(self):
        if self.Image is None:
            return
        sobel_x = self.Sobel(self.sobel_x_filter)
        sobel_y = self.Sobel(self.sobel_y_filter)

        combine_image = np.sqrt(np.square(sobel_x)+np.square(sobel_y))

        min_value = np.min(combine_image)
        max_value = np.max(combine_image)
        combine_image = np.uint8(combine_image / (max_value - min_value) * 255)

        return combine_image


    def GradientAngle(self):
        if self.Image is None:
            return
        sobel_x = self.Sobel(self.sobel_x_filter)
        sobel_y = self.Sobel(self.sobel_y_filter)

        gradient_angle = np.arctan2(sobel_y , sobel_x) * 180 / np.pi
        gradient_angle = (gradient_angle + 360) % 360

        mask1 = np.logical_and(gradient_angle >= 120, gradient_angle <= 180).astype(np.uint8) * 255
        mask2 = np.logical_and(gradient_angle >= 210, gradient_angle <= 330).astype(np.uint8) * 255

        # Apply masks to the combination image (assuming you already have the combination image)
        combination = self.CombineSobel()
        result1 = cv2.bitwise_and(combination, combination, mask=mask1)
        result2 = cv2.bitwise_and(combination, combination, mask=mask2)

        cv2.imshow('Result 1 (120-180 degrees)', result1)
        cv2.imshow('Result 2 (210-330 degrees)', result2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 4. Transforms
    def Transform(self):
        if self.Image is None:
            return
        image = cv2.imread(self.Image)
        cv2.imshow('Input Image', image)

        org_center = (240, 200)

        Rotate = float(self.ui.lineEdit_Rotation.text())
        Scaling = float(self.ui.lineEdit_Scaling.text())
        Tx = float(self.ui.lineEdit_Tx.text())
        Ty = float(self.ui.lineEdit_Ty.text())

        matrix = cv2.getRotationMatrix2D(org_center, Rotate, Scaling)
      #  result_image = cv2.warpAffine(image, RotationAndScaling_M,(image.shape[1], image.shape[0]))
        # Apply translation to the rotation matrix
        matrix[0, 2] += Tx
        matrix[1, 2] += Ty
        result_image = cv2.warpAffine(image, matrix,(image.shape[1], image.shape[0]))

        cv2.imshow('Output Image', result_image)












    '''
       # gradient_angle = np.arctan2(sobel_x , sobel_y) * 180 / np.pi
        gradient_angle = (gradient_angle + 360) % 360

        mask1 = np.logical_and(gradient_angle >= 120, gradient_angle <= 180).astype(np.uint8) * 255
        mask2 = np.logical_and(gradient_angle >= 210, gradient_angle <= 330).astype(np.uint8) * 255
        cv2.imshow('mask1', mask1)
        cv2.imshow('mask2', mask2)

        # Apply masks to the combination image (assuming you already have the combination image)
        combination = self.CombineSobel()
        result1 = cv2.bitwise_and(combination, combination, mask=mask1)
        result2 = cv2.bitwise_and(combination, combination, mask=mask2)

        cv2.imshow('Result 1 (120-180 degrees)', result1)
        cv2.imshow('Result 2 (210-330 degrees)', result2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

here


      
        image = cv2.imread(self.Image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Create an empty output image
        gradient_angle = np.zeros_like(smooth_image)
        # Iterate through the image and apply the Sobel y operator
        for y in range(1, smooth_image.shape[0] - 1):
            for x in range(1, smooth_image.shape[1] - 1):
                sobel_x_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * self.sobel_x_filter)
                sobel_y_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * self.sobel_y_filter)
                gradient_angle[y, x] = np.arctan2(sobel_y_pixel,sobel_x_pixel) * 180 / np.pi  # the pixel value at position (y,x)
                gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360
                
                
                
                
                
                
          gradient_angle = np.zeros_like(sobel_x)
        mask1 = np.zeros_like(sobel_x)
        mask2 = np.zeros_like(sobel_x)

        for y in range(1, sobel_x.shape[0] - 1):
            for x in range(1, sobel_x.shape[1] - 1):
                gradient_angle[y, x] = np.arctan2(sobel_x[y,x],sobel_y[y,x]) * 180 / np.pi  # the pixel value at position (y,x)
                gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360

        print(gradient_angle)

        mask1 = np.logical_and(gradient_angle >= 120, gradient_angle <= 180).astype(np.uint8) * 255
        mask2 = np.logical_and(gradient_angle >= 210, gradient_angle <= 330).astype(np.uint8) * 255
        cv2.imshow('mask1', mask1)
        cv2.imshow('mask2', mask2)

        # Apply masks to the combination image (assuming you already have the combination image)
        combination = self.CombineSobel()
        result1 = cv2.bitwise_and(combination, combination, mask=mask1)
        result2 = cv2.bitwise_and(combination, combination, mask=mask2)

        cv2.imshow('Result 1 (120-180 degrees)', result1)
        cv2.imshow('Result 2 (210-330 degrees)', result2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
                
                
                
                
                
                




        sobel_x_image = self.Sobel_Func(self.sobel_x_filter)
        sobel_y_image = self.Sobel_Func(self.sobel_y_filter)
    
        # Calculate the gradient angle using the cosine theorem
        dot_product = sobel_x_image * self.sobel_x_filter + sobel_y_image * self.sobel_y_filter
        magnitude_x = np.sqrt(sobel_x_image * sobel_x_image)
        magnitude_y = np.sqrt(sobel_y_image * sobel_y_image)
        cos_gradient_angle = dot_product / (magnitude_x * magnitude_y)
    
        # Handle cases where the cos_gradient_angle is out of the valid range [-1, 1]
        cos_gradient_angle = np.clip(cos_gradient_angle, -1, 1)
    
        # Calculate the gradient angle in radians
        gradient_angle_radians = np.arccos(cos_gradient_angle)
    
        # Convert gradient angle to degrees
        gradient_angle = np.degrees(gradient_angle_radians)
    
        # Ensure angles are between 0 and 360 degrees
        gradient_angle = (gradient_angle + 360) % 360
    
        print(gradient_angle)
    
        mask1 = np.logical_and(gradient_angle >= 120, gradient_angle <= 180).astype(np.uint8) * 255
        mask2 = np.logical_and(gradient_angle >= 210, gradient_angle <= 330).astype(np.uint8) * 255
    
        cv2.imshow('mask1', mask1)
        cv2.imshow('mask2', mask2)
    '''
    '''
        magnitude_x = np.abs(sobel_x_image)
        magnitude_y = np.abs(sobel_y_image)
    
        gradient_angle = np.arctan2(magnitude_y, magnitude_x) * 180 / np.pi
        gradient_angle = (gradient_angle + 360) % 360
        print(gradient_angle)
    
    
    
    
    
    
    
        # Calculate the dot product of Sobel x and Sobel y
        dot_product = sobel_x_image * sobel_y_image
    
        # Calculate the cosine of the gradient angle
        cos_gradient_angle = dot_product / (magnitude_x * magnitude_y)
    
        # Calculate the gradient angle in radians
        gradient_angle_rad = np.arccos(cos_gradient_angle)
    
        # Convert radians to degrees
        gradient_angle_deg = np.degrees(gradient_angle_rad)
    
        # Adjust the angle to the range [0, 360)
        gradient_angle_deg = (gradient_angle_deg + 360) % 360
    
        print(gradient_angle_deg)
    '''
        # Calculate gradient angle (in degrees) from Sobel x and Sobel y
     #   gradient_angle = np.degrees(np.arctan2(sobel_x_image, sobel_y_image))
    #    gradient_angle = (gradient_angle + 360) % 360
       # gradient_angle = (np.arctan2(sobel_x_image,sobel_y_image) * 180) / np.pi
       # gradient_angle = (gradient_angle + 360) % 360
     #   print(gradient_angle)

        # Define the angle ranges
      #  angle_range1 = (120, 180)  # 120 degrees to 180 degrees
     #   angle_range2 = (210, 330)  # 210 degrees to 330 degrees

      #  mask1 = cv2.inRange(gradient_angle, 120, 180)
     #   mask2 = cv2.inRange(gradient_angle, 210, 330)


        # mask1 0~120 = 0 ; 120~180 = 255 ; 180~360 = 0
     #   mask1 = cv2.threshold(gradient_angle ,119,255,cv2.THRESH_BINARY)  # 0~119 = 0 ; 120 ~360 = 255
     #   mask1_final = cv2.threshold(mask1,180,255,cv2.THRESH_TOZERO_INV) # 180~360 = 0 ; else won't change (0~120 = 0 ; 120 ~ 180 = 255)
     #   mask2 = cv2.threshold(gradient_angle ,209,255,cv2.THRESH_BINARY)  # 0~209 = 0 ; 210 ~360 = 255
     #   mask2_final = cv2.threshold(mask2,330,255,cv2.THRESH_TOZERO_INV) # 330~360 = 0 ; else won't change (0~209 = 0 ; 210 ~ 330 = 255)


        # Generate masks for specific angle ranges
     #   mask1 = np.logical_and(gradient_angle >= 120, gradient_angle <= 180).astype(
    #        np.uint8) * 255
    #    mask2 = np.logical_and(gradient_angle >= angle_range2[0], gradient_angle <= angle_range2[1]).astype(
    #        np.uint8) * 255
     #    print(gradient_angle)
     #    cv2.imshow('Angle', gradient_angle)

     #   mask1 = np.uint8(((gradient_angle >= 120) & (gradient_angle <= 180))) * 255
     #   mask2 = np.uint8(((gradient_angle >= 210) & (gradient_angle <= 330))) * 255
      #  cv2.imshow('mask1', mask1)
     #   cv2.imshow('mask2', mask2)
    #

        # Apply masks to the combination image (assuming you already have the combination image)
     #   combination = self.CombineSobel()
     #   result1 = cv2.bitwise_and(combination, combination, mask=mask1)
     #   result2 = cv2.bitwise_and(combination, combination, mask=mask2)

        # Display both results
    #    cv2.imshow('Result 1 (120-180 degrees)', result1)
     #   cv2.imshow('Result 2 (210-330 degrees)', result2)

     #   cv2.waitKey(0)
     #   cv2.destroyAllWindows()








