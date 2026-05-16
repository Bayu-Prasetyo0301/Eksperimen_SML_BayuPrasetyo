# =========================================================
# AUTOMATE PREPROCESSING
# =========================================================

# import library
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)

import joblib

print('=' * 50)
print('AUTOMATE PREPROCESSING DIMULAI')
print('=' * 50)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    '../BC-Churn_raw/Churn_Modelling.csv'
)

print('\nDataset berhasil dimuat')
print(df.shape)

# =========================================================
# DROP COLUMN
# =========================================================

df = df.drop(
    ['RowNumber', 'CustomerId', 'Surname'],
    axis=1
)

print('\nKolom tidak relevan berhasil dihapus')

# =========================================================
# LABEL ENCODING
# =========================================================

le = LabelEncoder()

df['Gender'] = le.fit_transform(
    df['Gender']
)

print('\nLabel Encoding berhasil')

# =========================================================
# ONE HOT ENCODING
# =========================================================

df = pd.get_dummies(
    df,
    columns=['Geography'],
    drop_first=True,
    dtype=int
)

print('\nOne Hot Encoding berhasil')

# =========================================================
# SPLIT FEATURE DAN TARGET
# =========================================================

X = df.drop('Exited', axis=1)
y = df['Exited']

print('\nFeature dan target berhasil dipisah')

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('\nTrain test split berhasil')

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\nFeature scaling berhasil')

# =========================================================
# SAVE SCALER
# =========================================================

joblib.dump(
    scaler,
    'scaler.pkl'
)

print('\nScaler berhasil disimpan')

# =========================================================
# SAVE PREPROCESSED DATASET
# =========================================================

processed_df = pd.DataFrame(
    X,
    columns=X.columns
)

processed_df['Exited'] = y.values

processed_df.to_csv(
    'Churn_Modelling_preprocessing.csv',
    index=False
)

print('\nDataset preprocessing berhasil disimpan')

# =========================================================
# FINAL OUTPUT
# =========================================================

print('=' * 50)
print('PREPROCESSING SELESAI')
print('=' * 50)

print('\nShape dataset akhir:')
print(processed_df.shape)