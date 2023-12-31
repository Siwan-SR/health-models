import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.preprocessing import PolynomialFeatures

def medical_cost():
    st.title("Medical Cost Prediction")

    st.write("Find what your medical cost will be.")

    age = st.slider("Age", 0, 130, 65)
    gender = st.number_input("Gender (Male: 0; Female: 1)")
    smoke = st.number_input("Do you smoke? (Yes: 1; No: 0)")
    bmi = st.number_input("BMI")
    children = st.number_input("How many children do you have?")
    region_northeast = st.number_input("Are you in the northeast? (Yes: 1, No: 0)")
    region_northwest = st.number_input("Are you in the northwest? (Yes: 1, No: 0)")
    region_south_east = st.number_input("Are you in the southeast? (Yes: 1, No: 0)")
    region_south_west = st.number_input("Are you in the southwest? (Yes: 1, No: 0)")


    model = pk.load(open("models/medical-cost-prediction.pkl", "rb"))

    submit = st.button("Predict!")

    if submit:
        X = [[age, gender, bmi, children, smoke, region_northeast, region_northwest, region_south_east, region_south_west]]
        poly = PolynomialFeatures()
        poly.fit(X)
        X = poly.transform(X)
        prediction = model.predict(X)
        prediction = round(int(prediction))
        st.write(f"Estimated medical cost: ${str(prediction)}")


