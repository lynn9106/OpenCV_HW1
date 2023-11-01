from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog

class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(600, 750)
        mainWindow.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")

    # Load Image
        self.btn_LoadImage = QtWidgets.QPushButton(self.centralwidget)
        self.btn_LoadImage.setGeometry(QtCore.QRect(20, 250, 80, 20))
        self.btn_LoadImage.setObjectName("btn_LoadImage")

        self.label_ImageName = QtWidgets.QLabel(self.centralwidget)
        self.label_ImageName.setGeometry(QtCore.QRect(25, 290, 90, 20))
        self.label_ImageName.setText("")
        self.label_ImageName.setObjectName("label_ImageName")


    # 1. Image Processing
        self.ImageProcessBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageProcessBox.setGeometry(QtCore.QRect(140, 40, 160, 180))
        self.ImageProcessBox.setAutoFillBackground(True)
        self.ImageProcessBox.setObjectName("ImageProcessBox")

        self.btn1_CSeperation = QtWidgets.QPushButton(self.ImageProcessBox)
        self.btn1_CSeperation.setGeometry(QtCore.QRect(20, 30, 120, 20))
        self.btn1_CSeperation.setObjectName("btn1_CSeperation")

        self.btn1_CTransformation = QtWidgets.QPushButton(self.ImageProcessBox)
        self.btn1_CTransformation.setGeometry(QtCore.QRect(15, 80, 130, 20))
        self.btn1_CTransformation.setObjectName("btn1_CTransformation")

        self.btn1_CExtraction = QtWidgets.QPushButton(self.ImageProcessBox)
        self.btn1_CExtraction.setGeometry(QtCore.QRect(20, 130, 120, 20))
        self.btn1_CExtraction.setObjectName("btn1_CExtraction")

    # 2. Image Smoothing
        self.ImageSmoothingBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ImageSmoothingBox.setGeometry(QtCore.QRect(140, 250, 160, 180))
        self.ImageSmoothingBox.setAutoFillBackground(True)
        self.ImageSmoothingBox.setObjectName("ImageSmoothingBox")

        self.btn2_Gaussian = QtWidgets.QPushButton(self.ImageSmoothingBox)
        self.btn2_Gaussian.setGeometry(QtCore.QRect(20, 30, 120, 20))
        self.btn2_Gaussian.setObjectName("btn2_Gaussian")

        self.btn2_Bilateral = QtWidgets.QPushButton(self.ImageSmoothingBox)
        self.btn2_Bilateral.setGeometry(QtCore.QRect(20, 80, 120, 20))
        self.btn2_Bilateral.setObjectName("btn2_Bilateral")

        self.btn2_Median = QtWidgets.QPushButton(self.ImageSmoothingBox)
        self.btn2_Median.setGeometry(QtCore.QRect(20, 130, 120, 20))
        self.btn2_Median.setObjectName("btn2_Median")

    # 3. Edge Detection
        self.EdgeDetectionBox = QtWidgets.QGroupBox(self.centralwidget)
        self.EdgeDetectionBox.setGeometry(QtCore.QRect(140, 460, 160, 250))
        self.EdgeDetectionBox.setAutoFillBackground(True)
        self.EdgeDetectionBox.setObjectName("EdgeDetectionBox")

        self.btn3_SobelX = QtWidgets.QPushButton(self.EdgeDetectionBox)
        self.btn3_SobelX.setGeometry(QtCore.QRect(20, 30, 120, 20))
        self.btn3_SobelX.setObjectName("btn3_SobelX")

        self.btn3_SobelY = QtWidgets.QPushButton(self.EdgeDetectionBox)
        self.btn3_SobelY.setGeometry(QtCore.QRect(20, 80, 120, 20))
        self.btn3_SobelY.setObjectName("btn3_SobelY")

        self.btn3_CombAndThres = QtWidgets.QPushButton(self.EdgeDetectionBox)
        self.btn3_CombAndThres.setGeometry(QtCore.QRect(20, 130, 120, 40))
        self.btn3_CombAndThres.setObjectName("btn3_CombAndThres")

        self.btn3_Gradient = QtWidgets.QPushButton(self.EdgeDetectionBox)
        self.btn3_Gradient.setGeometry(QtCore.QRect(20, 200, 120, 20))
        self.btn3_Gradient.setObjectName("btn3_Gradient")

    # 4. Transforms
        self.TransformsBox = QtWidgets.QGroupBox(self.centralwidget)
        self.TransformsBox.setGeometry(QtCore.QRect(340, 40, 201, 250))
        self.TransformsBox.setAutoFillBackground(True)
        self.TransformsBox.setObjectName("TransformsBox")

        self.btn4_Transforms = QtWidgets.QPushButton(self.TransformsBox)
        self.btn4_Transforms.setGeometry(QtCore.QRect(40, 200, 120, 20))
        self.btn4_Transforms.setObjectName("btn4_Transforms")

        self.label4_Rotation = QtWidgets.QLabel(self.TransformsBox)
        self.label4_Rotation.setGeometry(QtCore.QRect(10, 30, 45, 11))
        self.label4_Rotation.setObjectName("label4_Rotation")

        self.label4_Scaling = QtWidgets.QLabel(self.TransformsBox)
        self.label4_Scaling.setGeometry(QtCore.QRect(10, 70, 45, 11))
        self.label4_Scaling.setObjectName("label4_Scaling")

        self.label4_Tx = QtWidgets.QLabel(self.TransformsBox)
        self.label4_Tx.setGeometry(QtCore.QRect(10, 110, 45, 11))
        self.label4_Tx.setObjectName("label4_Tx")

        self.label4_Ty = QtWidgets.QLabel(self.TransformsBox)
        self.label4_Ty.setGeometry(QtCore.QRect(10, 150, 45, 11))
        self.label4_Ty.setObjectName("label4_Ty")

        self.label4_deg = QtWidgets.QLabel(self.TransformsBox)
        self.label4_deg.setGeometry(QtCore.QRect(160, 30, 21, 11))
        self.label4_deg.setObjectName("label4_deg")

        self.label4_TxPixel = QtWidgets.QLabel(self.TransformsBox)
        self.label4_TxPixel.setGeometry(QtCore.QRect(160, 110, 31, 11))
        self.label4_TxPixel.setObjectName("label4_TxPixel")

        self.label4_TyPixel = QtWidgets.QLabel(self.TransformsBox)
        self.label4_TyPixel.setGeometry(QtCore.QRect(160, 150, 31, 11))
        self.label4_TyPixel.setObjectName("label4_TyPixel")

        self.lineEdit_Rotation = QtWidgets.QLineEdit(self.TransformsBox)
        self.lineEdit_Rotation.setGeometry(QtCore.QRect(60, 25, 90, 20))
        self.lineEdit_Rotation.setObjectName("lineEdit_Rotation")

        self.lineEdit_Scaling = QtWidgets.QLineEdit(self.TransformsBox)
        self.lineEdit_Scaling.setGeometry(QtCore.QRect(60, 65, 90, 20))
        self.lineEdit_Scaling.setObjectName("lineEdit_Scaling")

        self.lineEdit_Tx = QtWidgets.QLineEdit(self.TransformsBox)
        self.lineEdit_Tx.setGeometry(QtCore.QRect(60, 105, 90, 20))
        self.lineEdit_Tx.setObjectName("lineEdit_Tx")

        self.lineEdit_Ty = QtWidgets.QLineEdit(self.TransformsBox)
        self.lineEdit_Ty.setGeometry(QtCore.QRect(60, 145, 90, 20))
        self.lineEdit_Ty.setObjectName("lineEdit_Ty")

    # 5. VGG19
        self.VGG19Box = QtWidgets.QGroupBox(self.centralwidget)
        self.VGG19Box.setGeometry(QtCore.QRect(340, 320, 201, 390))
        self.VGG19Box.setAutoFillBackground(True)
        self.VGG19Box.setObjectName("VGG19Box")

        self.btn5_LoadImage = QtWidgets.QPushButton(self.VGG19Box)
        self.btn5_LoadImage.setGeometry(QtCore.QRect(40, 20, 120, 20))
        self.btn5_LoadImage.setObjectName("btn5_LoadImage")

        self.btn5_AgumentedImages = QtWidgets.QPushButton(self.VGG19Box)
        self.btn5_AgumentedImages.setGeometry(QtCore.QRect(40, 55, 120, 40))
        self.btn5_AgumentedImages.setObjectName("btn5_AgumentedImages")

        self.btn5_ModelStruct = QtWidgets.QPushButton(self.VGG19Box)
        self.btn5_ModelStruct.setGeometry(QtCore.QRect(30, 110, 140, 20))
        self.btn5_ModelStruct.setObjectName("btn5_ModelStruct")

        self.btn5_AccAndLoss = QtWidgets.QPushButton(self.VGG19Box)
        self.btn5_AccAndLoss.setGeometry(QtCore.QRect(40, 150, 120, 20))
        self.btn5_AccAndLoss.setObjectName("btn5_AccAndLoss")

        self.btn5_Inference = QtWidgets.QPushButton(self.VGG19Box)
        self.btn5_Inference.setGeometry(QtCore.QRect(40, 190, 120, 20))
        self.btn5_Inference.setObjectName("btn5_Inference")

        self.label5_Predict = QtWidgets.QLabel(self.VGG19Box)
        self.label5_Predict.setGeometry(QtCore.QRect(40, 220, 100, 11))
        self.label5_Predict.setObjectName("label5_Predict")
        mainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "MainWindow"))
        self.btn_LoadImage.setText(_translate("mainWindow", "Load Image"))
        self.label_ImageName.setText(_translate("mainWindow", "No Image Loaded"))
        self.ImageProcessBox.setTitle(_translate("mainWindow", "1. Image Processing"))
        self.btn1_CSeperation.setText(_translate("mainWindow", "1.1 Color Seperation"))
        self.btn1_CTransformation.setText(_translate("mainWindow", "1.2 Color Transformation"))
        self.btn1_CExtraction.setText(_translate("mainWindow", "1.3 Color Extraction"))
        self.ImageSmoothingBox.setTitle(_translate("mainWindow", "2. Image Smoothing"))
        self.btn2_Gaussian.setText(_translate("mainWindow", "2.1 Gaussian blur"))
        self.btn2_Bilateral.setText(_translate("mainWindow", "2.2 Bilateral filter"))
        self.btn2_Median.setText(_translate("mainWindow", "2.3 Median filter"))
        self.EdgeDetectionBox.setTitle(_translate("mainWindow", "3. Edge Detection"))
        self.btn3_SobelX.setText(_translate("mainWindow", "3.1 Sobel X"))
        self.btn3_SobelY.setText(_translate("mainWindow", "3.2 Sobel Y"))
        self.btn3_CombAndThres.setText(_translate("mainWindow", "3.3 Combination and\n"
" Threshold"))
        self.btn3_Gradient.setText(_translate("mainWindow", "3.4 Gradient Angle"))
        self.TransformsBox.setTitle(_translate("mainWindow", "4. Transforms"))
        self.btn4_Transforms.setText(_translate("mainWindow", "4. Transforms"))
        self.label4_Rotation.setText(_translate("mainWindow", "Rotation:"))
        self.label4_Scaling.setText(_translate("mainWindow", "Scaling:"))
        self.label4_Tx.setText(_translate("mainWindow", "Tx:"))
        self.label4_Ty.setText(_translate("mainWindow", "Ty:"))
        self.label4_deg.setText(_translate("mainWindow", "deg"))
        self.label4_TxPixel.setText(_translate("mainWindow", "pixel"))
        self.label4_TyPixel.setText(_translate("mainWindow", "pixel"))
        self.VGG19Box.setTitle(_translate("mainWindow", "5. VGG19"))
        self.btn5_LoadImage.setText(_translate("mainWindow", "Load Image"))
        self.btn5_AgumentedImages.setText(_translate("mainWindow", "5.1 Show Agumented \n"
"Images"))
        self.btn5_ModelStruct.setText(_translate("mainWindow", "5.2 Show Model Structure"))
        self.btn5_AccAndLoss.setText(_translate("mainWindow", "5.3 Show Acc and Loss"))
        self.btn5_Inference.setText(_translate("mainWindow", "5.4 Inference"))
        self.label5_Predict.setText(_translate("mainWindow", "Predict ="))





