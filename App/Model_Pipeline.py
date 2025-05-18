import joblib
import pandas as pd
import numpy as np

model = joblib.load('../Model/lgbm.pkl')
encoder = joblib.load('../Model/onehot_encoder.pkl')
scaler = joblib.load('../Model/scaler.pkl')
columns = joblib.load('../Model/columns.pkl')

def preprocess_input(raw_df):
    raw_df['bed_bath_ratio'] = raw_df.apply(lambda x: x['beds'] / x['bath'] if x['bath'] != 0 else 0, axis=1)

    for col in ['propertysqft', 'bath', 'beds', 'bed_bath_ratio']:
        raw_df[col] = np.log1p(raw_df[col])

    return raw_df

def predict_price(raw_df):
    X = preprocess_input(raw_df)
    log_pred = model.predict(X)
    price_pred = np.expm1(log_pred)
    return price_pred

