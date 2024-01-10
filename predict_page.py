import streamlit as st
#from joblib import dump, load
import numpy as np
import pickle
import import_ipynb
from idk import my_pipeline

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']

def show_predict_page():
    st.title("Apple Stock Predictor")

    st.write("""### We need some information to predict price""")

    open = st.number_input("Enter Opening Value of Stock")
    high = st.number_input("Enter Highest Value of Stock")
    low = st.number_input("Enter Lowest Value of Stock")
    vol = st.number_input("Enter the Volume of Stock Bought")

    ok = st.button("Calculate Closing Value")
    if ok:
        from sklearn.preprocessing import StandardScaler
        std = StandardScaler()  


        x = np.array([[open, high, low, vol]])
        ig = my_pipeline.transform(x)
        #print(ig)
        close = regressor.predict(ig)
        st.subheader(f"The Estimated Stock value is {close[0]:.3f}")