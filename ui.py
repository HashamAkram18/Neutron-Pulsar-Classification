import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.prediction import predict_pulsar
from utils.pca_reduction import plot_3d_decision_boundary

# Define a function to handle the prediction page
def prediction_page():
    st.title("Neutron Star - Pulsar Prediction")

    # Input features
    st.sidebar.header("Input Features")
    mean_profile = st.sidebar.number_input("Mean of the integrated profile", value=0.0)
    sd_profile = st.sidebar.number_input("Standard deviation of the integrated profile", value=0.0)
    kurtosis_profile = st.sidebar.number_input("Excess kurtosis of the integrated profile", value=0.0)
    skew_profile = st.sidebar.number_input("Skewness of the integrated profile", value=0.0)
    mean_curve = st.sidebar.number_input("Mean of the DM-SNR curve", value=0.0)
    sd_curve = st.sidebar.number_input("Standard deviation of the DM-SNR curve", value=0.0)
    kurtosis_curve = st.sidebar.number_input("Excess kurtosis of the DM-SNR curve", value=0.0)
    skew_curve = st.sidebar.number_input("Skewness of the DM-SNR curve", value=0.0)

    # Load your saved model and scaler
    scaler = joblib.load("objects/scaler_ob.joblib")
    model = joblib.load("objects/rb_model.joblib")

    # Prediction button
    if st.sidebar.button("Predict"):
        input_data = pd.DataFrame({
            "Mean of the integrated profile": [mean_profile],
            "Standard deviation of the integrated profile": [sd_profile],
            "Excess kurtosis of the integrated profile": [kurtosis_profile],
            "Skewness of the integrated profile": [skew_profile],
            "Mean of the DM-SNR curve": [mean_curve],
            "Standard deviation of the DM-SNR curve": [sd_curve],
            "Excess kurtosis of the DM-SNR curve": [kurtosis_curve],
            "Skewness of the DM-SNR curve": [skew_curve]
        })

        prediction = predict_pulsar(model, scaler, input_data)

        if prediction == 0:
            st.write("Prediction: **Neutron Star**")
        else:
            st.write("Prediction: **Pulsar**")

# Define a function to handle the 3D decision boundary page
def pca_3d_page():
    st.title("PCA Decision Boundary")

    # Load PCA model and object
    model = joblib.load('objects/rb_model.joblib')  # Assuming this is your model
    pca_ob = joblib.load('objects/pca_ob.joblib')  # Load the correct PCA object

    # Show the 3D decision boundary
    plot_3d_decision_boundary(model, pca_ob)  # Make sure the function takes the right parameters

# Main function to handle page navigation
def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Prediction", "3D Decision Boundary"])

    # Display the selected page
    if page == "Prediction":
        prediction_page()
    elif page == "3D Decision Boundary":
        st.title("3D PCA Reduction Decision Boundary for Neutron Star vs Pulsar Classification")
        
        # Description for the decision boundary
        st.write("""
        This visualization represents the decision boundary for classifying neutron stars and pulsars using Principal Component Analysis (PCA) to reduce the data's dimensionality. 
        By transforming the original multi-dimensional feature space into three principal components, we can plot the decision surface of the classifier in a more interpretable 3D space. 
        The decision boundary helps illustrate how the model distinguishes between the two classes (neutron stars and pulsars) based on the transformed features, providing an intuitive understanding of the classification process after dimensionality reduction.
        """)
        if st.button("Show 3D Decision Boundary"):
            pca_3d_page()

if __name__ == "__main__":
    main()
