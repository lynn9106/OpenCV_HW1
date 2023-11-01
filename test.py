import cv2
import numpy as np

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


