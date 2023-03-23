import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import load_img

#from keras.applications.imagenet_utils import preprocess_input

upload_img = st.file_uploader('Insert image for classification', type=['png', 'jpg'])

imgs_model_width, imgs_model_height = 224, 224

if upload_img is not None:
    img = Image.open(upload_img)
    #numpy_image = np.asarray(img)
    #image_batch = np.expand_dims(numpy_image, axis=0)
    #processed_image = preprocess_input(image_batch.copy())
    st.image(img)
