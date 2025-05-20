# 🏠 New York House Price Prediction

## 📌 Table of Contents

- [Technologies Used](#technologies-used)
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
- [Data Preprocessing and Analysis](#data-preprocessing-and-analysis)
- [Model Training and Selection](#model-training-and-selection)
- [Streamlit App Implementation](#streamlit-app-implementation)
- [How to Run](#how-to-run)
- [Lessons Learned](#lessons-learned)
- [Demo](#demo)
- [License](#license)

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Data Manipulation**: pandas, numpy, scipy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning Libraries**: scikit-learn, lightgbm
- **Models Used**: Linear Regression, K-Nearest Neighbors Regression, Random Forest, LightGBM
- **Environment**: Jupyter Notebook
- **Model Deployment**: Streamlit
- **Model Persistence**: joblib

---

## 📊 Project Overview

The New York House Price Prediction project aims to build a machine learning model that can accurately estimate house prices based on various features such as property type, location, number of bedrooms and bathrooms, square footage, and more.  

This project involves:
- Exploring and analyzing real estate data collected from different locations in New York.
- Cleaning and preprocessing the data to handle missing values and inconsistencies.
- Training and evaluating several regression models, including Linear Regression, KNN Regression, Random Forest, and LightGBM.
- Deploying the final model as a user-friendly web application using Streamlit, allowing users to input house features and get instant price predictions.

The goal is to demonstrate practical skills in data analysis, feature engineering, model selection, and deployment.

---

## 🗂️ Dataset Overview

- **Source**: https://www.kaggle.com/code/bbjadeja/new-york-houseprediction/input  
- **Size**: 4801 rows, 17 columns  
- **Main features**:  
  - `PROPERTYSQFT`: Total square footage of the property  
  - `BEDS` & `BATH`: Number of bedrooms and bathrooms  
  - `LOCALITY` & `SUBLOCALITY`: Geographical location of the house
  - `LATITUDE` & `LONGITUDE`: Geographic coordinates used to determine the exact location of the property, which can significantly influence its price based on proximity to amenities, central areas, or specific neighborhoods.

---

## 📖 Data Preprocessing and Analysis



---

## 📊 EDA (Exploratory Data Analysis)

> Summarize the insights you found during EDA

- Correlation between features
- Distribution of target variable
- Outliers & missing values
- Feature importance (if applicable)

---

## 🤖 Modeling

| Model              | RMSE / R² Score | Notes                       |
|-------------------|-----------------|-----------------------------|
| Linear Regression | `___`           | `e.g., baseline model`      |
| Random Forest     | `___`           | `e.g., better generalization`|
| XGBoost           | `___`           | `e.g., highest performance` |

---

## 📈 Results

- Best performing model: `__________`
- Key metrics:
  - MAE: `___`
  - RMSE: `___`
  - R² Score: `___`

---

## ▶️ How to Run

```bash
# Clone the repo
git clone https://github.com/your-username/your-repo-name.git

# Navigate to the project folder
cd your-repo-name

# Run the notebook
Open the .ipynb file in Jupyter or Colab
