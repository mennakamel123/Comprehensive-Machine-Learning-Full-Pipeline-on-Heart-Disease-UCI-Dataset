# Comprehensive-Machine-Learning-Full-Pipeline-on-Heart-Disease-UCI-Dataset
AI_ML Final Project of Sprints X Microsoft
# Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset
## 1. Project Overview

This project aims to analyze, predict, and visualize heart disease risks using machine
learning. The workflow involves data preprocessing, feature selection, dimensionality
reduction (PCA), model training, evaluation, and deployment. Classification models like
Logistic Regression, Decision Trees, Random Forest, and SVM will be used, alongside
K-Means and Hierarchical Clustering for unsupervised learning. Additionally, a Streamlit UI
will be built for user interaction, deployed via Ngrok, and the project will be hosted on GitHub.

---

## 2. Key Features

- **Data Cleaning:** Handles missing values and prepares the dataset for analysis.
- **Exploratory Data Analysis (EDA):** Visualizes data distributions and feature correlations.
- **Feature Engineering:** Includes dimensionality reduction with PCA and feature selection with Chi-Square, Random Forest Importance, and RFE.
- **Supervised Learning:** Trains and evaluates four classification models:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
- **Unsupervised Learning:** Explores natural data clusters using K-Means and Hierarchical Clustering.
- **Model Optimization:** Enhances the best-performing model using `GridSearchCV` for hyperparameter tuning.
- **Interactive UI:** A web application built with Streamlit allows users to input their health data and receive real-time predictions.
- **Deployment:** The local web application is made publicly accessible using Ngrok.

---

## 3. File Structure
```
Heart_Disease_Project/
│
├── data/
│   ├── heart_disease.csv
│   ├── cleaned_dataset.csv
│   ├── pca_transformed_dataset.csv
│   └── feature_selected_dataset.csv
│
├── deployment/
│   └── ngrok_setup.txt
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── results/
│   └── evaluation_metrics.md
│
├── ui/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
---
```
## 4. Setup and Installation

To run this project locally, please follow these steps:

**Prerequisites:**
- Python 3.8 or higher
- Pip (Python package installer)

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/alshafeay/Heart_Disease.git
    cd Heart_Disease_Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. How to Use the Project

### Running the Analysis Scripts

You can run the analysis steps sequentially using the Python scripts or Jupyter Notebooks provided. For example, to run the data preprocessing script:
```bash
python 01_data_preprocessing.py
Running the Web Application
Start the Streamlit App:
Make sure you are in the root Heart_Disease_Project directory.

Bash

streamlit run ui/app.py
This will open the application in your local browser.

Deploy with Ngrok (Optional):
To share the app, open a second terminal and run:

Bash

ngrok http 8501
Use the public URL provided by Ngrok to access the app from anywhere.

6. Model Summary
The final model selected for deployment was a Tuned Random Forest Classifier. After hyperparameter optimization, it achieved strong performance metrics, proving to be the most robust model for this classification task. For a detailed breakdown of all model results, please see the results/evaluation_metrics.md file.

7. Tools and Libraries
Python

Pandas & NumPy: For data manipulation.

Matplotlib & Seaborn: For data visualization.

Scikit-learn: For machine learning modeling and preprocessing.

Streamlit: For building the interactive web UI.

Ngrok: For deploying the local web application.

Jupyter Notebook: For exploratory analysis.
