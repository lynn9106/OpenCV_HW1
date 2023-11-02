import sys

import cv2
import numpy as np
import math

# Load the image
image = cv2.imread(r'C:\Users\lynn9106\PycharmProjects\OpenCV_HW1\Q3_image\building.jpg')

'''
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

'''


def Sobel_Func(filter):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    height, width = smooth_image.shape

    sobel_image = np.zeros((height, width), dtype=np.int32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            #sobel_pixel = 0
            #for i in range (-1, 2):
            #    for j in range (-1, 2):
            #        sobel_pixel = sobel_pixel + smooth_image[y+i,x+j] * filter[1+i,1+j]
            sobel_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * filter)
            sobel_image[y, x] = np.sqrt(np.square(sobel_pixel))  # the pixel value at position (y,x)

    # normalize between [0,255]
    min_value = np.min(sobel_image)
    max_value = np.max(sobel_image)

    sobel_image = np.uint8(sobel_image / (max_value - min_value) * 255)

    return sobel_image

'''
        image = cv2.imread(self.Image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)


        gradient_angle = np.zeros_like(sobel_x)
        for y in range(0, sobel_x.shape[0]):
            for x in range(0, sobel_x.shape[1]):
              #  gradient_angle[y, x] = np.arctan2(sobel_y[y,x],sobel_x[y,x]) * 180 / np.pi  # the pixel value at position (y,x)
              #  gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360
                sobel_x_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * self.sobel_x_filter)
                sobel_x_t = np.sqrt(np.square(sobel_x_pixel))
                print(sobel_x[y,x])
                print(sobel_x_t)
                print()


print("sobel:")
print(sobel_pixel)
print(smooth_image[y + i, x + j])
print(filter[1 + i, 1 + j])

print("end")
'''

  #          if(sobel_image[y, x]<min):
   #             min = sobel_image[y, x]
  #          if(sobel_image[y,x]>max):
  #              max = sobel_image[y,x]


    # normalize between [0,255]
   # min_value = np.min(sobel_image)
  #  max_value = np.max(sobel_image)
  #  print(min)
 #   print(max)
  #  print(min_value)
 #   print(max_value)
 #   print()
  #  sobel_image = np.uint8(sobel_image / (max_value - min_value) * 255)


# Sobel x filter for edge detection
sobel_x_filter = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
# Sobel y filter for edge detection
sobel_y_filter = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

sobel_x = Sobel_Func(sobel_x_filter)
sobel_y = Sobel_Func(sobel_y_filter)

height, width = sobel_x.shape
gradient_angle = np.zeros((height, width), dtype=np.int32)


for y in range(0, height):
    for x in range(0, width):
        print(sobel_y[y,x])
        print(sobel_x[y,x])
        gradient_angle[y, x] = math.atan2(sobel_y[y, x], sobel_x[y, x]) * 180 / math.pi
        gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360
        print(gradient_angle[y, x])
        print()


#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
'''
sobel_x = Sobel_Func(sobel_x_filter)
sobel_y = Sobel_Func(sobel_y_filter)

gradient_angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
gradient_angle = (gradient_angle + 360) % 360
print(gradient_angle)



for y in range(0, height):
    for x in range(0, width):
        gradient_angle[y, x] = np.arctan2(sobel_y[y, x], sobel_x[y, x]) * 180 / np.pi
        gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360
        print(gradient_angle[y, x])


gradient_angle = np.zeros_like(sobel_x)
for y in range(1, sobel_x.shape[0]-1):
    for x in range(1, sobel_x.shape[1]-1):
        #  gradient_angle[y, x] = np.arctan2(sobel_y[y,x],sobel_x[y,x]) * 180 / np.pi  # the pixel value at position (y,x)
        #  gradient_angle[y, x] = (gradient_angle[y, x] + 360) % 360
        sobel_x_pixel = np.sum(smooth_image[y - 1:y + 2, x - 1:x + 2] * sobel_x_filter)
        print(sobel_x_pixel)
        sobel_x_t = np.sqrt(np.square(sobel_x_pixel))
        print(sobel_x[y, x])
        print(sobel_x_t)
        print()





def sobel_edge_detection(self):
    # Define the Sobel operators
    Gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    if self.Image is None:
        return
    image = cv2.imread(self.Image)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    smooth_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Get the dimensions of the image
    height, width = smooth_image.shape
    # Create an output image with the same dimensions
    out_image = np.zeros((height, width), dtype=np.int32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                out_image[i, j] = 0
            else:
                X = 0
                Y = 0
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        X += Gx[1 + k, 1 + l] * np.int32(smooth_image[i + k, j + l])
                        Y += Gy[1 + k, 1 + l] * np.int32(smooth_image[i + k, j + l])
                out_image[i, j] = np.abs(X) + np.abs(Y)

    # Normalize the output image to the range [0, 255]
    max_value = np.max(out_image)
    min_value = np.min(out_image)
    out_image = np.uint8(np.round(out_image / (max_value - min_value) * 255))

    # Display the Sobel x image
    cv2.imshow('Sobel X Image', out_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''

