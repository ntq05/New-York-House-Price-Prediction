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
- [Reflections and Limitations](#reflections-and-limitations)
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

The goal of this project is to apply the knowledge and techniques acquired from Data Mining, Statistical Methods, and Statistical Learning courses to a real-world problem, bridging theory and practical implementation.

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

---

## 🤖 Model Training and Selection

### Models Used:
- **Linear Regression**
- **K-Nearest Neighbors (KNN) Regression**
- **Random Forest Regression**
- **LightGBM (LGBM) Regression**

### Model Selection Strategy

#### 1. Linear Regression & KNN Regression
- Applied **cross-validation** to evaluate model performance.
- Evaluation metrics: **R²**, **MAE**, **MSE**, **RMSE**.
- **Linear Regression** was not suitable due to the lack of linearity in the dataset.
- For **KNN Regression**, used **Grid Search Cross-Validation** to find the optimal hyperparameters based on R², MAE, and MSE.
- The selected KNN models were then evaluated on the test set using the same metrics.

#### 2. Random Forest & LightGBM
- Due to high computational cost, **Randomized Search Cross-Validation** was used instead of Grid Search.
- Metrics used for model selection: **R²**, **MAE**, **MSE**.
- Built 3 candidate models for each algorithm based on randomized search results.
- Evaluated each model on the test set using **R²**, **MAE**, **MSE**, and **RMSE**, and selected the best-performing one.

#### 3. Feature Importance
- Used **feature importance** analysis from Random Forest and LGBM models.
- Helped identify which features had the most impact on the predictions.

---

### Final Model Selection
- **Best Model:** LightGBM (Model 2 - optimized for MAE)
- **Selection Method:** Randomized Search CV (based on MAE)
- **Performance on Test Set:**

| Model               | R²       | MAE           | MSE             | RMSE           |
|--------------------|----------|---------------|------------------|----------------|
| LGBM Model 2 (MAE) | 0.580829 | 711,646.09    | 1.056096 × 10¹³  | 3.249762 × 10⁶ |

---

## 🌐 Streamlit App Implementation

To provide an interactive interface for predicting house prices in New York, a **Streamlit web application** was developed. The app allows users to input property details and instantly receive a predicted price using the trained **LightGBM model**.

### Key Components:

- **Model Loading**:
  - Loaded the pre-trained LGBM model (`lgbm.pkl`) along with its preprocessing components:
    - One-hot encoder (`onehot_encoder.pkl`)
    - Scaler (`scaler.pkl`)
    - Column structure (`columns.pkl`)

- **Preprocessing Function**:
  - Added a new feature `bed_bath_ratio` to enhance model performance.
  - Applied log transformation to features: `beds`, `bath`, `propertysqft`, and the derived `bed_bath_ratio`.
  
- **Prediction Function**:
  - Performed preprocessing on user input.
  - Used the model to make predictions in log scale, and converted the result back to the original price scale using `np.expm1`.

### App Features:

- User-friendly input form for:
  - Number of beds
  - Number of bathrooms
  - Property square footage
  - Latitude
  - Longitude

- On clicking the **"Predict"** button:
  - User input is transformed into a DataFrame.
  - Prediction is made and displayed in dollar format with two decimal places.

This implementation provides a clean and interactive interface for end-users to experiment with different property configurations and receive real-time price estimations.

## 🚀 How to Run the Project

### Step 1: Set Up the Environment

1. Install [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.
2. Open your terminal or command prompt and navigate to the directory that contains the `environment.yml` file.
3. Create the conda environment by running: conda env create -f environment.yml
4. After the environment is created, activate it: conda activate NYHousePrediction

### Step 2: Run the Jupyter Notebook

Once the environment is activated, you can launch Jupyter Notebook.
Open the notebook file in your browser and execute the cells to explore data and model training.

### Step 3: Run the Streamlit App
1. Navigate to the App directory: cd App
2. Launch the Streamlit application: streamlit run app.py
3. The app will open in your browser. If not, go to the URL shown in your terminal

--- 

## 📚 Lessons Learned

- Applied **statistical methods** for exploratory data analysis (EDA) and data preprocessing to better understand and prepare the dataset.
- Implemented various **machine learning models** learned from coursework in **statistical learning** and **machine learning** classes.
- Gained hands-on experience with new techniques and tools:
  - **Lorenz Curve**: Used to identify the optimal thresholds for re-categorizing data.
  - **Matplotlib & Seaborn**: Learned how to visualize data effectively and interpret various types of plots.

These lessons helped deepen both my theoretical understanding and practical skills in data science and model development.

---

## 🔄 Reflections and Limitations

While this project provided a valuable opportunity to apply knowledge from Data Mining, Statistical Methods, and Statistical Learning, there were several limitations:

- The dataset had notable inconsistencies and lacked important features that could have improved prediction performance.
- As this was my first end-to-end machine learning project, there were areas where optimization techniques and advanced preprocessing could have been better utilized.
- Model performance, although functional, did not reach ideal levels — this highlights the importance of domain understanding and data quality.

Nonetheless, the process greatly enhanced my practical understanding of the ML pipeline and revealed areas for continued improvement in future projects.


## 📄 License

This project is created for educational and portfolio purposes only.  
All rights are reserved by the author. Unauthorized use, reproduction, or distribution of any part of this project is prohibited without explicit permission.

© 2025 Thiện Quân Nguyễn

