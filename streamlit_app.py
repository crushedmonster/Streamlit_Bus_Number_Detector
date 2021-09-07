# -*- coding: utf-8 -*-
"""
@author: Wenna
"""

# import libraries
import torch
import numpy as np
from PIL import Image
import pytesseract
from gtts import gTTS
import streamlit as st
from common import * # available in the common.py file

##########
##### Set up page title and icon.
##########

# page title
st.set_page_config(page_title = "Bus Number Detector", 
    page_icon=":bus:")


##########
##### Set up sidebar.
##########

st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

st.sidebar.write('[Find additional images here.](https://github.com/crushedmonster)')
st.sidebar.write('<br>', unsafe_allow_html=True)
st.sidebar.write('### :floppy_disk: Parameters')
st.sidebar.write('#### Inference Setting:')
st.sidebar.write('<br>', unsafe_allow_html=True)

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the \
                                         minimum acceptable confidence level \
                                             for displaying a bounding box?', 
                                             0.0, 1.0, 0.5, 0.01)

st.sidebar.write('<br>', unsafe_allow_html=True)
st.sidebar.write('<br>', unsafe_allow_html=True)
st.sidebar.write('<br>', unsafe_allow_html=True)

st.sidebar.write(':desktop_computer: [Source code in Github](https://github.com/crushedmonster/Streamlit_Bus_Number_Detector)')


##########
##### Set up main app.
##########
# include web app title and description
st.title('Bus Number Detector :bus:')
st.write('This web app made use of a combination of Object Detection (using YOLOv5) \
             and Optical Character Recognition (OCR), to extract bus number from \
                 bus panel and convert the extracted text to audio.')

# include instructions
st.write("""
         **Instructions:**
         """)
st.write("""
         👈 Feel free to upload any image you want on the side bar.
         """)
st.write("""
         :camera: Scroll down to view demo.
         """)
                              
## pull in default image or user-selected image.
if uploaded_file is None:
    # default image.
    img = './bus_sample_images/bus_video1_198.jpg'
    image = Image.open(img)

else:
    # user-selected image
    image = Image.open(uploaded_file)
    
# convert to numpy array
image = np.array(image)
image2 = image.copy()

## Object Detection
# model
model = torch.hub.load('yolov5', 'custom', path='best', source='local') 

# get bounding box
st.write('### Inferenced Image')

# inference settings
model.conf = float(f'{confidence_threshold}')  # confidence threshold (0-1)
results = model(image2, size=640) # custom inference size

# results
# display image.
st.image(results.render(),
         use_column_width=True)

## get bounding box
bounding_box = results.pandas().xyxy[0]  # img predictions (pandas)

## check for any bounding box
if len(bounding_box) == 0:
    st.write('### Result')
    st.write('No bus number detected.')
    
else:
    # grab ROI
    # xmin
    x_min = int(bounding_box['xmin'][0])
    # xmax
    x_max = int(bounding_box['xmax'][0])
    # ymin
    y_min = int(bounding_box['ymin'][0])
    # ymax
    y_max = int(bounding_box['ymax'][0])
    
    # use numpy slicing to crop the region of interest
    roi = image[y_min:y_max,x_min:x_max]
    
    # get grayscale image
    gray = get_grayscale(roi)
    # apply thresholding
    thresh = thresholding(gray)

    ## OCR
    # OCR the input image using Tesseract
    # recognise only digits by changing the config to the following
    custom_config = r'--oem 3 --psm 6 outputbase digits'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    st.write('### Result')
    # check if output is a number
    try:
        number = int(text)
    except ValueError:
        st.write("Could not read the bus number.")
    else:
        # output the number if detected
        st.write(f'Bus Number: {int(text)}')
    
        ## Convert Text to Audio
        audio_text = f'Bus {int(text)}'
        language = 'en'
        audio_obj = gTTS(text=audio_text, lang=language, slow=False) 
        audio_obj.save("./bus_number.mp4") 
        
        # Display option to play audio file
        audio_file = open('bus_number.mp4', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/ogg')