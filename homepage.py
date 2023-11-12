import streamlit as st
from diabetes import *
from lung_cancer import *

def home():
    st.sidebar.success("Select a model above.")

    st.title("Welcome to Health Models")
    st.subheader("Find out whether you have a disease with excellent health models.")
    st.write("Find models in GitHub page.")


if __name__ == "__main__":
    home()

if st.sidebar.checkbox("Diabetes", False):
    diabetes()

if st.sidebar.checkbox("Lung Cancer", False):
    lung_cancer()