'''
This program will demonstrate the basics of image filtering using OpenCV
and numpy

'''

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('cv_logo.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # this is necessary because openCV uses BGR format and matplotlib reads images using RGB format


kernel = np.ones((5, 5), np.float32) / 25  # kernel is defines by 1 / (K_width * K_height) so, for 5x5, must be 1/25 * 5x5 matrix of ones for homogeneous filter

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


# used to blur images, takes in image, and kernel
# blur = cv2.blur(img, (5, 5))
# titles = ['images', '2D Convolution', 'blur']
# images = [img, dst, blur]

# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks()
#     plt.yticks()
    
# plt.show()

# GAUSSIAN FILTER
# Guassian filter is just using different kernel in both x and y direction
# kernel matrix has highest value in the middle and decreases as we traverse outward

# img = cv2.imread('Halftone_Gaussian_Blur.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# kernel = np.ones((5, 5), np.float32) / 25
# dst = cv2.filter2D(img, -1, kernel)


# blur = cv2.blur(img, (5, 5))
# g_blur = cv2.GaussianBlur(img, (5, 5), 0)
# titles = ['images', '2D Convolution', 'blur', 'Gaussian Blur']
# images = [img, dst, blur, g_blur]

# for i in range(4):
#     plt.subplot(1, 4, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks()
#     plt.yticks()

# plt.show()

# MEDIAN FILTERING
# - replace each pixel's value with the median of its neighboring pixels.
# - this method is great when dealing with "salt and pepper" noise
# img = cv2.imread('salt_and_pepper.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# kernel = np.ones((5, 5), np.float32) / 25
# dst = cv2.filter2D(img, -1, kernel)


# blur = cv2.blur(img, (5, 5))
# g_blur = cv2.GaussianBlur(img, (5, 5), 0)
# median = cv2.medianBlur(img, 5)  # kernel must be odd except for 1

# titles = ['images', '2D Convolution', 'blur', 'Gaussian Blur', 'Median Blur']
# images = [img, dst, blur, g_blur, median]

# for i in range(5):
#     plt.subplot(1, 5, i + 1)
#     plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks()
#     plt.yticks()

# plt.show()

# BILATERAL FILTER
# edges remian sharp even when blurred
img = cv2.imread('girl.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(img, -1, kernel)


blur = cv2.blur(img, (5, 5))
g_blur = cv2.GaussianBlur(img, (5, 5), 0)
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
titles = ['images', '2D Convolution', 'blur', 'Gaussian Blur', 'Bilateral']
images = [img, dst, blur, g_blur, bilateral]

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks()
    plt.yticks()

plt.show()


