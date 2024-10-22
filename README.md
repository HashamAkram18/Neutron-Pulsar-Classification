# Neutron Star - Pulsar Classification

![Project Logo](https://github.com/HashamAkram18/Neutron-Pulsar-Classification/blob/main/images/Screenshot%202024-10-22%20180740.png?raw=true) <!-- Placeholder for your project logo -->

## Overview

This project focuses on the classification of neutron stars and pulsars using machine learning techniques. The goal is to build a robust predictive model that can accurately distinguish between these two classes based on various features derived from astrophysical data. Leveraging advanced machine learning libraries, we achieved a high validation accuracy, demonstrating the effectiveness of our approach.

## Table of Contents

- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Modeling](#modeling)
- [Performance](#performance)
- [User Interface](#user-interface)
- [Visualization](#visualization)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Technologies Used

- Python
- Scikit-learn
- Optuna (Hyperparameter Optimization)
- Streamlit (User Interface)
- Plotly (Data Visualization)
- NumPy
- Pandas

## Dataset [Link](https://www.kaggle.com/datasets/colearninglounge/predicting-pulsar-starintermediate)


The dataset consists of various features related to neutron stars and pulsars, including statistical measures like mean, standard deviation, kurtosis, and skewness of the integrated profiles and DM-SNR curves. The dataset was preprocessed to handle missing values through mean imputation and to address class imbalance using class weights.

## Modeling

We experimented with various machine learning classifiers to identify the best-performing model for our classification task:

- **Gradient Boosting**: Accuracy - 0.976531
- **Logistic Regression**: Accuracy - 0.959090
- **Random Forest**: Accuracy - 0.971330
- **Support Vector Machine (SVM)**: Accuracy - 0.965875

The hyperparameter optimization was conducted using **Optuna**, which allowed us to fine-tune our models effectively.

![Optimization History Graph](https://github.com/HashamAkram18/Neutron-Pulsar-Classification/blob/main/images/newplot%20(4).png?raw=true)

### Best Model Performance

- **Validation Accuracy with Best Parameters**: **0.9785**

![AUC-ROC Thresholds](https://github.com/HashamAkram18/Neutron-Pulsar-Classification/blob/main/images/newplot%20(3).png?raw=true)

## User Interface

The user interface is built using **Streamlit**, providing an interactive experience for users to input features and receive predictions on whether a given object is a neutron star or a pulsar.

## Visualization

We utilized **Plotly** to create a 3D visualization of the PCA-reduced decision boundaries, helping to illustrate how the models differentiate between neutron stars and pulsars based on the transformed feature space.

![PCA Visualization](https://github.com/HashamAkram18/Neutron-Pulsar-Classification/blob/main/images/Screenshot%202024-10-22%20180835.png?raw=true) <!-- Placeholder for PCA visualization image -->
![](https://github.com/HashamAkram18/Neutron-Pulsar-Classification/blob/main/images/nut-pul.png?raw=true)
## Installation

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/neutron-pulsar-classification.git
   cd neutron-pulsar-classification
