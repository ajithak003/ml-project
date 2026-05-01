import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def clean_data(df):
   

   """Task 1 — Clean the Data Drop customerID, replace blank spaces in TotalCharges 
   with NaN, convert it to numeric, impute missing values with the column median, 
   and encode Churn as 1 (Yes) and 0 (No)."""

   df.drop('customerID', axis=1, inplace=True)
   
   df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
   df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
   df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())   
   df['Churn'] = (df['Churn'] == 'Yes').astype(int) 
   

def encode_scale_data(df):

    """TTask 2 — Encode, Split, and Scale One-hot encode gender and ContractType with drop_first=True, 
    perform a stratified 80/20 split with random_state=42, then fit a StandardScaler on training data 
    only and transform both sets."""

    df = pd.get_dummies(df, columns=['gender', 'ContractType'], drop_first=True)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn'] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
    
def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):

    """Task 3 — Train and Evaluate Train a LogisticRegression model with random_state=42 and max_iter=1000, 
    then print train and test accuracy."""

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f'Train Accuracy: {train_accuracy:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    return model

    
def model_threshold_tuning(model, X_test_scaled, y_test):

    """Task 4 — Threshold Tuning Using model.predict_proba(), print the number of test customers 
    flagged as churners at threshold 0.5 and at threshold 0.3, and comment on which a limited-budget 
    retention team should prefer."""

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    for threshold in [0.3, 0.5, 0.7]:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_threshold)
        print(f'Accuracy at threshold {threshold}: {accuracy:.4f}')




data = {
    'customerID':     ['C001','C002','C003','C004','C005','C006','C007','C008',
                       'C009','C010','C011','C012','C013','C014','C015','C016',
                       'C017','C018','C019','C020'],
    'gender':         ['Male','Female','Male','Female','Male','Female','Male','Female',
                       'Male','Female','Male','Female','Male','Female','Male','Female',
                       'Male','Female','Male','Female'],
    'SeniorCitizen':  [0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1],
    'tenure':         [12,34,2,45,1,22,8,60,15,3,27,50,7,18,42,5,30,55,10,24],
    'ContractType':   ['Month-to-month','One year','Month-to-month','Two year',
                       'Month-to-month','One year','Month-to-month','Two year',
                       'Month-to-month','Month-to-month','One year','Two year',
                       'Month-to-month','Month-to-month','Two year','Month-to-month',
                       'One year','Two year','Month-to-month','One year'],
    'MonthlyCharges': [70.5,55.0,85.3,42.0,95.1,60.0,78.4,38.5,88.0,72.0,
                       52.0,35.0,91.2,65.0,40.0,80.0,58.0,33.0,76.5,62.0],
    'TotalCharges':   ['846.0','1870.0',' ','1890.0','95.1','1320.0','627.2','2310.0',
                       '1320.0','216.0','1404.0','1750.0','638.4','1170.0','1680.0',
                       '400.0','1740.0','1815.0','765.0','1488.0'],
    'Churn':          ['Yes','No','Yes','No','Yes','No','Yes','No','Yes','Yes',
                       'No','No','Yes','Yes','No','Yes','No','No','Yes','No']
}

df = pd.DataFrame(data)

clean_data(df)
X_train_scaled, X_test_scaled, y_train, y_test = encode_scale_data(df)
model = train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
model_threshold_tuning(model, X_test_scaled, y_test)
