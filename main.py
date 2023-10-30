import sys
from PyQt5.QtWidgets import  QApplication,QMainWindow,QWidget ,QPushButton, QVBoxLayout,QLineEdit,QLabel
from PyQt5 import QtCore, QtGui


#定義按鍵動作用的函式
def buttonClick():
    print("Button Clicked")

# 設定按鈕樣式
style_btn = '''
    QPushButton{
        background:#ff0;
        border:1px solid #000;
        border-radius:10px;
        padding:5px;
    }
    QPushButton:pressed{
        background:#f90;
    }
'''
font = QtGui.QFont()
font.setPointSize(12)


app = QApplication(sys.argv) # close the window by sys

# Main Window
mainWindow = QWidget() #Create Window
mainWindow.resize(1000, 1200) #Resize Window
mainWindow.setWindowTitle("Hw1") # Set Title


#Button Load Image
#A vertical layout with two buttons
LoadImageBox = QWidget(mainWindow)
LoadImageBox.setGeometry(0,0,300,900)
#LoadImageBox.setStyleSheet(style_box)

LoadImage_layout = QVBoxLayout(LoadImageBox)
LoadImage_layout.setAlignment(QtCore.Qt.AlignVCenter)  # 垂直置中對齊

btn1_load = QPushButton("Load Image 1", LoadImageBox)
#btn1_load.setStyleSheet(style_btn)
LoadImage_layout.addWidget(btn1_load)
btn2_load = QPushButton("Load Image 2", LoadImageBox)
#btn2_load.setStyleSheet(style_btn)
LoadImage_layout.addWidget(btn2_load)

#Image Processing
ImageProcessingBox = QWidget(mainWindow)
ImageProcessingBox.setGeometry(300,0,300,600)
ImageProcessing_layout = QVBoxLayout(ImageProcessingBox)
ImageProcessing_layout.setAlignment(QtCore.Qt.AlignVCenter)
Label1 = QLabel(ImageProcessingBox)
Label1.setText('1.Image Processing')
Label1.setGeometry(0, 10, 280, 50)
Label1.setFont(font)                      # 文字大小
btn1_IProcess = QPushButton("1.1 Color Seperate", ImageProcessingBox)
btn1_IProcess.setGeometry(10, 70, 280, 50)
btn2_IProcess = QPushButton("1.2 Color Transform",ImageProcessingBox)
btn2_IProcess.setGeometry(10, 130, 280, 50)
btn3_IProcess = QPushButton("1.3 Color Extraction",ImageProcessingBox)
btn3_IProcess.setGeometry(10, 200, 280, 50)

#Image Smoothing







mainWindow.show() # 讓視窗顯現出來

sys.exit(app.exec_()) #sys.exit()當我們關閉程式時可以幫助我們離開


