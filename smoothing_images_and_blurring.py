'''
This program will demonstrate the basics of image filtering using OpenCV
and numpy

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cv_logo.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # this is necessary because openCV uses BGR format and matplotlib reads images using RGB format

# a kernel is a 
kernel = np.ones((5, 5), np.float32) / 25  # kernel is defines by 1 / (K_width * K_height) so, for 5x5, must be 1/25 * 5x5 matrix of ones for homogeneous filter (mean)

# arguments = image, desired depth of the destination image,  3rd is the kernel
dst = cv2.filter2D(img, -1, kernel)

titles = ['image', '2D Convolution']
images = [img, dst]

for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    
plt.show()

# the result should show the first image unfiltered, with the second image with less noise and a bit more blur

# 1 Dimensional singals can be filtered with Low Pass Filtering (LPF) and High Pass Filtering (HPF)
# LPF - helps in removing noise, blurring images
# HPF - helps in finding edges in the images



