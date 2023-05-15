# This file takes an image and convolves it with a list of filters.
import numpy as np
import cv2

if __name__ == '__main__':
    def convolve_image(image, filters):
        
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for filter in filters:    
            result = cv2.filter2D(image, ddepth=-1, kernel=filter)
            cv2.imshow("Filtered image", result)
            cv2.waitKey(0)
            cv2.destroyWindow("Filtered image")

#Box filter with kernel size 5*5
filter_1 = (1/25) * np.ones((5,5))

#Derivative filter
filter_2 = np.array([-5,0,5]).reshape((-1,3))

#Gaussian filter
def gaussian_filter():
 
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    sigma = int(input("Enter the value of sigma: "))
    muu = int(input("Enter the value of mean, muu: "))
    kernel_size = 2*3*sigma+1
    x, y = np.meshgrid(np.linspace(-3*sigma, 3*sigma, kernel_size),
                       np.linspace(-3*sigma, 3*sigma, kernel_size))
    dst = ((x-muu)**2+(y-muu)**2)/(2*sigma**2)
 
    # lower normal part of gaussian
    normal = 1/(2 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    return np.exp(-dst) * normal

#Gaussian filter
filter_3 = gaussian_filter()

#Sharpening filter
filter_4 = np.array([[0,-1,0], [0,3.5,0], [0,-1,0]])

# filters = [filter_1, filter_2, filter_3, filter_4]
filters = [filter_1, filter_2, filter_3, filter_4]

# image = 'C:\RWTH Aachen University\Semester 4\CV Practice\Data files\lena.jpg'
image = input("Enter the file path.\nThe file path should not have inverted brackets: ")
convolve_image(image, filters)
