# -*- coding: utf-8 -*-
"""
@author: Wenna
This python file holds all the common functions used.
"""
import cv2
import numpy as np
import pytesseract

## Image Processing
# define helper function to process ROI

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,3)

# thresholding
def thresholding(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# OCR Setup
my_env = "cloud"
if my_env == "cloud": ## <-- TOGGLE THIS
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
else: # running on local
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'