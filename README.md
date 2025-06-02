# 🏠 New York House Price Prediction

A data science project that predicts house prices in New York using features such as property type, broker information, number of bedrooms and bathrooms, square footage, and detailed location data including latitude, longitude, and various address components.

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

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Data Manipulation**: pandas, numpy, scipy
- **Data Visualization**: matplotlib, seaborn
- **Machine Learning Libraries**: scikit-learn, lightgbm
- **Models Used**: Linear Regression, K-Nearest Neighbors Regression, Random Forest, LightGBM
- **Environment**: Jupyter Notebook
- **Model Deployment**: Streamlit
- **Model Persistence**: joblib

## 📊 Project Overview

The New York House Price Prediction project aims to build a machine learning model that can accurately estimate house prices based on various features such as property type, location, number of bedrooms and bathrooms, square footage, and more.  

This project involves:
- Exploring and analyzing real estate data collected from different locations in New York.
- Cleaning and preprocessing the data to handle missing values and inconsistencies.
- Training and evaluating several regression models, including Linear Regression, KNN Regression, Random Forest, and LightGBM.
- Deploying the final model as a user-friendly web application using Streamlit, allowing users to input house features and get instant price predictions.

The goal is to demonstrate practical skills in data analysis, feature engineering, model selection, and deployment.

## 📂 Dataset Overview

The dataset used in this project was sourced from [Kaggle](https://www.kaggle.com/code/bbjadeja/new-york-houseprediction/input) and contains detailed information about residential properties in New York.

- **Source**: [Kaggle - New York House Prediction](https://www.kaggle.com/code/bbjadeja/new-york-houseprediction/input)
- **Size**: 4,801 rows × 17 columns
- **Data Type**: Tabular data including both numerical and categorical features
- **Target Variable**: `PRICE` – the selling price of the property

**Main features used for prediction include:**
- `PROPERTYSQFT`: Total square footage of the property
- `BEDS` & `BATH`: Number of bedrooms and bathrooms
- `LOCALITY` & `SUBLOCALITY`: Location information at the neighborhood level
- `LATITUDE` & `LONGITUDE`: Geographic coordinates indicating the property's exact position, which can impact price depending on proximity to desirable areas
- `TYPE`: Type of property (e.g., Condo, Apartment, House)
- `BROKERTITLE`: Title or role of the real estate broker

The dataset required several preprocessing steps such as handling missing values, encoding categorical features, and selecting relevant variables to improve model performance.

## 📖 Data Preprocessing and Analysis

### 1. Splitting the Dataset into Training and Test Sets  
To prevent data leakage and ensure a fair evaluation of the model's performance, the dataset was divided into training and test sets.

### 2. Univariate and Bivariate Analysis  

#### 🔹 Univariate Analysis  
Univariate analysis focuses on understanding the distribution and characteristics of each feature individually.

- **For numerical features**, the following statistical metrics were examined:
  - **Mean**: Indicates the central tendency of the data  
  - **Standard deviation**: Measures the dispersion of data points from the mean  
  - **Minimum and maximum values**  
  - **Median, 25th percentile (Q1), and 75th percentile (Q3)**  
  - **Kurtosis**: Provides insight into the presence of outliers by measuring tail heaviness  
  - **Skewness**: Describes the asymmetry of the distribution  
  - **Data type**  
  - **Number of missing values**

- **For categorical features**, the analysis included:
  - **Number of unique values**  
  - **Mode**: Identifies the most frequent category  
  - **Data type** (e.g., `object`, `datetime`)  
  - **Number of missing values**

To address class imbalance in categorical features, a **cumulative sum method** was applied. Classes with low frequency were grouped under an `'Others'` category based on a threshold determined using the **Gini coefficient** and **Lorenz curve**.

**Visualizations**:
- **Categorical features**: Count plots were used to visualize class distributions  
- **Numerical features**: Histograms were used to observe data distribution, skewness, and potential outliers

#### 🔹 Bivariate Analysis  
The goal of bivariate analysis is to understand the relationship between each feature and the target variable.

- **Numerical predictors**:  
  - **Pearson correlation** was used to detect linear relationships with the numeric target

- **Categorical predictors**:  
  - **Mann–Whitney U test** (instead of t-test)  
  - **Kruskal–Wallis test** (instead of ANOVA)  
These non-parametric tests were selected due to the non-normal distribution of the target variable, which violates the assumptions of t-tests and ANOVA.

**Visualizations**:
- **Numerical predictors**: Line graphs and correlation heatmaps  
- **Categorical predictors**: Mean bar plots and box plots were used to compare the distribution of the target across different classes

The results of the statistical tests helped assess whether observed differences were **statistically significant**.

---

### 3. Feature Engineering and Noteworthy Preprocessing Steps  

- **Property Type (`type`)**:
  - The feature included ambiguous or status-related values such as `'For sale'`, `'Pending'`, `'Contingent'`, `'Foreclosure'`, etc.  
  - `'For sale'` was treated as missing and imputed with the mode (`'Co-op for sale'`)  
  - `'Pending'`, `'Contingent'`, and `'Foreclosure'` were identified as property status rather than types.  
    A new feature called `status` was created to store these values, and the original `type` entries were imputed similarly

- **State**:
  - The `state` feature had values in the format `<borough>, NY <zip code>`  
  - A new feature `borough` was extracted, and the original `state` column was removed

- **Bed-to-Bathroom Ratio**:
  - A new feature `bed_bath_ratio` was created to analyze the balance between bedrooms and bathrooms

---

### 4. Multivariate Analysis  

- **Pair plots** and **correlation heatmaps** were used to identify **multicollinearity** between numerical features  
- Weak linear relationships with the target prompted **log transformation** on selected highly skewed, right-tailed features

---

### 5. Final Preprocessing Steps  

- **One-hot encoding** was applied to all categorical features  
- **RobustScaler** was used for normalization. This method is robust to outliers, as it scales features using the median and interquartile range (IQR), making it more suitable for this dataset than standard scaling

