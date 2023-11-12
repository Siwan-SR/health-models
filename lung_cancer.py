import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def lung_cancer():
    st.title("Lung Cancer Prediction")

    st.write("You may search up any category according to age and gender.")
    st.write("Please write 2 if condition is True. Please write 1 if condition is False")

    gender = st.number_input("Gender (Male: 1; Female: 0)")
    age = st.slider("Age", 0, 130, 65)
    smoke = st.number_input("Do you smoke?")
    yellow = st.number_input("Do you have yellow fingers?")
    anxiety = st.number_input("Do you suffer from anxiety?")
    peer = 1
    chronic = st.number_input("Do you have a chronic lung disease?")
    fatigue = st.number_input("Do you have fatigues regularly?")
    allergy = st.number_input("Do you have any allergies?")
    wheeze = st.number_input("Do you wheeze regularly but unintentionally?")
    alcohol = st.number_input("Do you drink alcohol often?")
    cough = st.number_input("Do you cough regularly?")
    breath = st.number_input("Do you suffer from shortness of breath?")
    swallow = st.number_input("Do you have problem from swallowing?")
    pain = st.number_input("Do you suffer from chest pain?")
    outcome = st.number_input("Do you think you have lung cancer?")

    # Gender: M(male): 1, F(female): 0
    # Age: Age of the patient
    # Smoking: YES = 2, NO = 1.
    # Yellow fingers: YES = 2, NO = 1.
    # Anxiety: YES = 2, NO = 1.
    # Peer_pressure: YES = 2, NO = 1.
    # Chronic Disease: YES = 2, NO = 1.
    # Fatigue: YES = 2, NO = 1.
    # Allergy: YES = 2, NO = 1.
    # Wheezing: YES = 2, NO = 1.
    # Alcohol: YES = 2, NO = 1.
    # Coughing: YES = 2, NO = 1.
    # Shortness of Breath: YES = 2, NO = 1.
    # Swallowing Difficulty: YES = 2, NO = 1.
    # Chest pain: YES = 2, NO = 1.
    # Lung Cancer: YES: 1, NO: 0.


    model = pk.load(open("models/lung-cancer-detection-model.pkl", "rb"))

    submit = st.button("Predict!")

    if submit:
        X = [[gender, age, smoke, yellow, anxiety, peer, chronic, fatigue, allergy, wheeze, alcohol, cough, breath, swallow, pain]]
        y = [outcome]

        prediction = model.score(X, y)

        if prediction == 0:
            st.write("Patient does not have lung cancer.")

        else:
            st.write("Patient has lung cancer.")


lung_cancer()