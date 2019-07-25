'''
Author: Jonilyn Tejada-Dabalos
Version: 1.0
Date Created: July 25, 2019
Summary: This code removes an image backround and returns a grayscaled foreground using Semantic Segmentation
Reference: https://www.learnopencv.com/applications-of-foreground-background-separation-with-semantic-segmentation
'''


import cv2
import numpy as np


image = cv2.imread("messi2.jpeg")


# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# create mask
background = 255 * np.ones_like(gray).astype(np.uint8)
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
alpha = cv2.GaussianBlur(threshed, (7,7),0)
	 
# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float)/255

#Convert foreground to float for multiplication
foreground = gray.astype(float)
background = background.astype(float)

# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)
	 
# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)
	 
# remove the background from the foreground
outImage = cv2.subtract(background, foreground)
	 
res = outImage/255

cv2.imshow("result", res)
cv2.imshow("original", image)
cv2.waitKey()
