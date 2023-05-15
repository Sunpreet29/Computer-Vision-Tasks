"""
This file takes an image as an input and returns it after applying edge detection algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # Read the image
    image = cv2.imread(input("Please enter the image file path: "))
    # Convert BGR to gray scale image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(image, (3,3), 0)

    horizontal_1 = np.hstack((image, blurred_image))
    
    #Second arg: lower threshold, third arg: upper threshold
    wide = cv2.Canny(blurred_image, 10, 200)
    mid = cv2.Canny(blurred_image, 30, 150)
    tight = cv2.Canny(blurred_image, 240, 250)

    horizontal_2 = np.hstack((wide, mid, tight))

    # vertical = np.vstack((horizontal_1, horizontal_2))
    cv2.imshow("Image1: high threshold difference, Image2: medium threshold difference, Image3: Low threshold difference", horizontal_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()