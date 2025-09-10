import streamlit as st
import pandas as pd
from ucimlrepo import fetch_ucirepo


st.set_page_config(
    page_title="Home Page",
    page_icon="🏠",
)

st.write("# IT5006 Group 9 Project")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    Our group performed Exploratory Data Analysis (EDA) on the project dataset from the UCI Machine Learning Repository.
    **👈 Select a page from the sidebar** to view the different sections.
    ### Group Members
    - Agnesh Kumar Rai
    - Aletheia Lai Xuanyu
    - Dan Yi Jia
"""
)

@st.cache_data
def load_data():
    """Fetches the Diabetes dataset from the UCI repository and returns it as a DataFrame."""
    diabetes_dataset = fetch_ucirepo(id=296)
    X = diabetes_dataset.data.features
    y = diabetes_dataset.data.targets
    df = pd.concat([X, y], axis=1)
    return df