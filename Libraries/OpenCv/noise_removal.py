import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#builtiin noise removal functions
# cv2.fastNlMeansDenoising() - works for single grayscale images
# cv2.fastNlMeansDenoisingColored() - works for colored images
# cv2.fastNlMeansDenoisingMulti() - works for image sequence of grayscale images
# cv2.fastNlMeansDenoisingColoredMulti() - works for image sequence of colored images

# Read an image
img = cv.imread('Resources/Photos/cat.jpg')
cv.imshow('Cat', img)

dst = cv.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
cv.imshow('Cat Denoised', dst)
cv.waitKey(0)