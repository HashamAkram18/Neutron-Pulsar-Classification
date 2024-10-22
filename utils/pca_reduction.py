import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def plot_3d_decision_boundary(model, pca):
    """
    Plots the 3D decision boundary for a trained model using PCA.

    Args:
        model: The trained classification model.
        pca: The PCA object used for dimensionality reduction.
    """
    # Load data
    X_train_scaled = joblib.load("utils\X_train_scaled.joblib")
    y_train = joblib.load("utils\y_train.joblib")

    X_train_pca = pca.fit_transform(X_train_scaled)

    # Fit the model with the reduced data
    model.fit(X_train_pca, y_train)

    # Create a meshgrid for the 3D plot
    xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min(), X_train_pca[:, 0].max(), 100),
                         np.linspace(X_train_pca[:, 1].min(), X_train_pca[:, 1].max(), 100))

    # Predict on the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])
    Z = Z.reshape(xx.shape)

    # Create the 3D plot using Plotly
    fig = go.Figure()

    # Add decision boundary surface
    fig.add_trace(go.Surface(z=Z, x=xx, y=yy, opacity=0.5, colorscale='Viridis', name='Decision Boundary'))

    # Plot the data points
    fig.add_trace(go.Scatter3d(
        x=X_train_pca[:, 0],
        y=X_train_pca[:, 1],
        z=X_train_pca[:, 2],
        mode='markers',
        marker=dict(size=5, color=y_train, colorscale='Viridis', line=dict(width=0.5)),
        name='Data Points'
    ))

    # Update layout to set white background and black axis lines
    fig.update_layout(
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Principal Component 3',
            xaxis=dict(
                backgroundcolor='#e0f6fc',  # Set the background of the x-axis to white
                color='black',            # Set the axis label color to black
                showbackground=True,      # Show axis background
                gridcolor='black'         # Set the grid color to black
            ),
            yaxis=dict(
                backgroundcolor='#e0f6fc',  # Set the background of the y-axis to white
                color='black',            # Set the axis label color to black
                showbackground=True,      # Show axis background
                gridcolor='black'         # Set the grid color to black
            ),
            zaxis=dict(
                backgroundcolor='#e0f6fc',  # Set the background of the z-axis to white
                color='black',            # Set the axis label color to black
                showbackground=True,      # Show axis background
                gridcolor='black'         # Set the grid color to black
            )
        ),
        paper_bgcolor='white',  # Set the paper background (outside the plot) to white
        plot_bgcolor='white'    # Set the plot area background to white
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

