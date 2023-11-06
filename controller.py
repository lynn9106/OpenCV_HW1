from MainUI import Ui_mainWindow

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap

from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19_bn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torchsummary
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
        self.ui.btn5_AgumentedImages.clicked.connect(self.AugmentedImage)
        self.ui.btn5_ModelStruct.clicked.connect(self.ModelStruct)
        self.ui.btn5_AccAndLoss.clicked.connect(self.AccAndLoss)
        self.ui.btn5_LoadImage.clicked.connect(self.LoadImage_5)
        self.ui.btn5_Inference.clicked.connect(self.Inference)


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

        average_weighted_image = (blue_channel + green_channel + red_channel) / 3
        average_weighted_image = average_weighted_image.astype(np.uint8)

        cv2.imshow('I2(Averaged Weighted)', average_weighted_image)
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

    def AugmentedImage(self):
        image_folder = './Q5_image/Q5_1/'

        augmented_images = []
        images_name = []

        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flip
            transforms.RandomVerticalFlip(),  # Random vertical flip
        ])
        transforms.RandomRotation(30),  # Random rotation (up to 30 degrees)

        for file in os.listdir(image_folder):       # Load and augment the images
            if file.endswith(".png"):
                file_path = os.path.join(image_folder , file)
                image = Image.open(file_path)

                augmented = data_transforms(image)
                augmented_images.append(augmented)
                images_name.append(file)

        # Display the augmented images
        fig, axes = plt.subplots(3, 3, figsize=(13, 13))

        i = 0
        for image in augmented_images:
            row = i // 3
            col = i % 3
            axes[row, col].imshow(image)
            axes[row, col].set_title(images_name[i])
            i += 1
        plt.show()

    def ModelStruct(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = vgg19_bn(num_classes=10)  # build a VGG19 with batch normalization (BN) model
        model.to(device)
        torchsummary.summary(model, (3, 224, 224))  # show the structure
    def AccAndLoss(self):
        image_path = './training_validation_plot.png'  # Replace with the actual file path
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def LoadImage_5(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '','Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)',options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.ui.image_label.setPixmap(pixmap.scaled(128, 128))
            pixmap.save('temp_image.png')  # Save the displayed image to a temporary file


    def Inference(self):

        # Check if a CUDA-compatible GPU is available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the class labels
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Define the image transformation
        inference_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Path to the image you want to infer
        image_path = 'temp_image.png'

        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' does not exist.")
        else:
            # Load and transform the image
            image = inference_transform(Image.open(image_path))
            image = image.unsqueeze(0)
            image = image.to(device)

            # Load the pre-trained model
            model = vgg19_bn(num_classes=10)  # Replace 'YourModel' with your actual model class
            model.load_state_dict(torch.load('best_model_weights.pth'))
            model.to(device)

            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                class_label = classes[predicted]
            #    print(f"Predicted Class: {class_label}")
            #    print(f"Class Probabilities: {probabilities[0]}")

                self.ui.label5_Predict.setText(f"Predict =  {class_label}")

                probability = probabilities[0].cpu().numpy()
                # Create a histogram of the prediction probabilities
                plt.figure(figsize=(8, 4))
                plt.bar(range(len(probability)), probability)
                plt.xlabel('Class')
                plt.ylabel('Probability')
                plt.title('Prediction Probability Distribution')
                plt.xticks(range(len(classes)), classes, rotation=45)
                plt.tight_layout()

                # Display the histogram in a new window
                plt.show()



