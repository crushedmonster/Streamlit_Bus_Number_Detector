# -*- coding: utf-8 -*-
"""
@author: Wenna
This python file holds all the common functions used.
"""
import cv2
import numpy as np
import pytesseract
import torch

## Image Processing
# define helper function to process ROI

# get grayscale image
@st.cache
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# thresholding
@st.cache
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# noise removal
@st.cache
def remove_noise(image):
    return cv2.medianBlur(image,3)

# opening - erosion followed by dilation
@st.cache
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
# Define function to load model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('yolov5', 'custom', path='best', source='local')
    return model

# OCR Setup
my_env = "cloud" ## <-- TOGGLE THIS
if my_env == "cloud": 
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
else: # running on local
    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'