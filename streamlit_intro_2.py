import streamlit as st
import numpy as np
import pandas as pd

st.title('North pole penguin')

penguin_df=pd.read_csv('dataset/penguins.csv.xls')
penguin_df

#st.button('click me')

def data_view():
    return st.write(data)
bt = st.button('click me',on_click=data_view)

#camera input
pic = st.camera_input('Take a pic')
if pic:
    img = image.open(pic)
    st.image(img)