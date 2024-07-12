import streamlit as st
import pandas as pd

df= pd.read_csv('dataset/iris.csv')

iris_dataset = st.multiselect(label='Features of Iris-Flower ',
                             options=[df.columns[0],df.columns[1],df.columns[2],df.columns[3],df.columns[4]],
                              default=df.columns[4])