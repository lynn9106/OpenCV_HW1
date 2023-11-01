import cv2
import numpy as np

# Load the image
image = cv2.imread(r'C:\Users\lynn9106\PycharmProjects\OpenCV_HW1\Q1_image\rgb.jpg')
zeros = np.zeros(image.shape[:2], dtype = "uint8")
# Split the BGR channels
blue_channel, green_channel, red_channel = cv2.split(image)

B_image = cv2.merge([blue_channel,zeros,zeros])
G_image = cv2.merge([zeros,green_channel,zeros])
R_image = cv2.merge([zeros,zeros,red_channel])

cv2.imshow('B Channel', B_image)
cv2.imshow('G Channel', G_image)
cv2.imshow('R Channel', R_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()


