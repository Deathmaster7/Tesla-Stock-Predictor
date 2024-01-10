import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from idk import stock_show
def show_explore_page():
    st.title("Apple Stock Dataset")
    st.dataframe(stock_show)