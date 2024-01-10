import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
opt = st.sidebar.selectbox("Explore or Predict", ("Predict", "Dataset"))

if opt == "Predict":
    show_predict_page()
if opt == "Dataset":
    show_explore_page()