from numpy import resize
import streamlit as st
import cv2 as cv
import numpy as np
from main import classifier

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('# Welcome to CNN image analyser')




img = st.file_uploader(label='Enter your image here', type=['jpg','jpeg','png'])

if img is not None:
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image.", width=350)
    st.markdown(f'## This is the image of a {classifier(image)}')
