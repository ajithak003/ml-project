import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def clean_data(df):
    """Task 1 — Clean the Data Replace blank spaces in TotalCharges with NaN, 
    convert it to float, drop rows with missing values, and encode the Churn 
    column as 1 (Yes) and 0 (No)."""


    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna().copy()  # Drop rows with NaN values and create a copy to avoid SettingWithCopyWarning
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

def encode_scale_data(df):
    """Task 2 — Preprocess and Split Encode ContractType using one-hot encoding 
    with drop_first=True. Perform a stratified 80/20 split with random_state=42, 
    then fit a StandardScaler on training data only and transform both sets."""

    df = pd.get_dummies(df, columns=['ContractType'], drop_first=True)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Task 3 — Train and Evaluate Train a LogisticRegression model with random_state=42. 
    Print train accuracy, test accuracy, precision, recall, and the confusion matrix."""

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return model


def rank_by_churn_probability(model, X_test_scaled, y_test):
    """Rank by Churn Probability Using predict_proba(), print the top 5 customers from 
    the test set ranked by highest churn probability, showing their index and probability. 
    In a comment, explain why ranking by probability is more useful than applying a fixed threshold 
    when the retention team has a limited call budget."""

    churn_probabilities = model.predict_proba(X_test_scaled)[:, 1]
    top_5_indices = np.argsort(churn_probabilities)[-5:][::-1]
    print(f"Top 5 indices by churn probability: {top_5_indices}")
    top_5_probabilities = churn_probabilities[top_5_indices]
    print(f"Top 5 churn probabilities: {top_5_probabilities}")

    print("Top 5 customers by churn probability:")
    for idx, prob in zip(top_5_indices, top_5_probabilities):
        print(f"Index: {idx}, Churn Probability: {prob:.4f}")

    


data = {
    'tenure':          [1, 34, 2, 45, 8, 22, 60, 3, 15, 50,
                        7, 18, 42, 5, 30, 55, 10, 24, 12, 40],
    'MonthlyCharges':  [95.1, 42.0, 85.3, 38.5, 78.4, 60.0, 35.0, 72.0,
                        88.0, 33.0, 91.2, 65.0, 40.0, 80.0, 58.0, 30.0,
                        76.5, 62.0, 70.5, 45.0],
    'TotalCharges':    ['95.1', '1428.0', ' ', '1732.5', '627.2', '1320.0',
                        '2100.0', '216.0', '1320.0', '1650.0', '638.4',
                        '1170.0', '1680.0', '400.0', '1740.0', '1650.0',
                        '765.0', '1488.0', '846.0', '1800.0'],
    'ContractType':    ['Month-to-month', 'One year', 'Month-to-month', 'Two year',
                        'Month-to-month', 'One year', 'Two year', 'Month-to-month',
                        'Month-to-month', 'Two year', 'Month-to-month', 'Month-to-month',
                        'Two year', 'Month-to-month', 'One year', 'Two year',
                        'Month-to-month', 'One year', 'Month-to-month', 'One year'],
    'Churn':           ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes',
                        'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No',
                        'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

df = clean_data(df)
X_train_scaled, X_test_scaled, y_train, y_test = encode_scale_data(df)
model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
rank_by_churn_probability(model, X_test_scaled, y_test)