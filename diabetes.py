import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


def diabetes():
    st.title("Diabetes Prediction")



    st.write("You may search up any category according to age and gender.")

    pregnancies = st.number_input("Number of Pregnancies")
    glucose = st.number_input("Glucose Level")
    pressure = st.number_input("Blood Pressure")
    skin = 0.01 # mode
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI (write in 1 decimal place)")
    function = 0.254 # mode
    age = st.number_input("Age")


    model = pk.load(open("models/diabetes-detection-model.pkl", "rb"))

    submit = st.button("Predict")

    if submit:
        X = [[pregnancies, glucose, pressure, skin, insulin, bmi, function, age]]
        prediction = model.predict(X)

        if prediction == 0:
            st.write("Patient does not have Diabetes.")

        else:
            st.write("Patient has Diabetes.")


diabetes()