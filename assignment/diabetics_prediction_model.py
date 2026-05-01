import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def clean_data(df):

    """Task 1 — EDA and Clean Invalid Zeros Print the shape, describe() summary, and class distribution of the Outcome 
    column. Replace zero values in BMI, Insulin, and BloodPressure with the column median. Confirm no zeros remain in 
    those columns."""
    
    df.describe()

    print(f"Class distribution of Outcome column:\n{df['Outcome'].value_counts()}")

    df['BMI'] = df['BMI'].replace(0, np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
    df['Insulin'] = df['Insulin'].replace(0, np.nan)

    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())    


    print(f"Number of zeros in BMI column: {df[df['BMI'] == 0].shape[0]}")
    print(f"Number of zeros in BloodPressure column: {df[df['BloodPressure'] == 0].shape[0]}")
    print(f"Number of zeros in Insulin column: {df[df['Insulin'] == 0].shape[0]}")

    return df


def preprocess_and_split(df):

    """Task 2 — Preprocess and Split Separate features (X) and target (y — Outcome). 
    Perform a stratified 80/20 split with random_state=42. Fit a StandardScaler on training data only 
    and transform both sets."""

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    
    """TTrain and Evaluate Train a LogisticRegression model with random_state=42 and 
    max_iter=1000. Print train accuracy, test accuracy, confusion matrix, precision, 
    and recall. In a comment, explain which metric matters most for this healthcare use case and why."""

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)
    class_report = classification_report(y_test, y_test_pred)

    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return model

    # In this healthcare use case, recall (sensitivity) is the most critical metric because it 
    # measures the model's ability to correctly identify patients with diabetes. Missing a 
    # positive case (false negative) could lead to a lack of necessary medical intervention,
    #  which can have severe consequences for the patient's health. Therefore, maximizing 
    # recall is essential to ensure that as many true positive cases as possible are identified.


def threshold_tuning(model, X_test, y_test):
    """Task 4 — Threshold Tuning Using predict_proba(), apply a threshold of 0.3 and recompute precision 
    and recall. In a comment, explain how the metrics changed and whether the lower threshold better serves 
    the clinical goal of minimising missed diabetes cases."""

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_threshold = (y_proba >= 0.3).astype(int)

    conf_matrix_threshold = confusion_matrix(y_test, y_pred_threshold)
    class_report_threshold = classification_report(y_test, y_pred_threshold)

    print("Confusion Matrix with Threshold 0.3:")
    print(conf_matrix_threshold)    
    print("Classification Report with Threshold 0.3:")
    print(class_report_threshold)

    # By lowering the threshold to 0.3, we are more likely to classify borderline cases as positive for diabetes, 
    # which can increase recall (sensitivity) but may decrease precision. This means that while we may identify 
    # more true positive cases (reducing missed diabetes cases), we may also have more false positives,
    # which could lead to unnecessary anxiety and medical interventions for patients who do not actually
    # have diabetes. Therefore, while the lower threshold may better serve the clinical goal of minimizing 
    # missed diabetes cases, it is important to balance this with the potential increase in false positives 
    # and consider the implications for patient care.


def main():

    data = {
        'Pregnancies':        [6, 1, 8, 1, 0, 5, 3, 10, 2, 8,
                               4, 10, 10, 1, 5, 7, 0, 7, 1, 1],
        'Glucose':            [148, 85, 183, 89, 137, 116, 78, 115, 197, 125,
                               110, 168, 139, 189, 166, 100, 118, 107, 103, 115],
        'BloodPressure':      [72, 66, 64, 66, 40, 74, 50, 0, 70, 96,
                              
                           92, 74, 80, 60, 72, 0, 84, 80, 30, 70],
        'BMI':                [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0,
                           37.6, 38.2, 27.1, 30.1, 25.8, 30.0, 27.3, 30.5, 27.5, 29.0],
        'Insulin':            [0, 0, 0, 94, 168, 0, 0, 0, 543, 0,
                           0, 0, 0, 125, 0, 0, 0, 0, 0, 0],
        'Age':                [50, 31, 32, 21, 33, 30, 26, 29, 53, 54,
                           31, 35, 57, 59, 51, 32, 33, 40, 27, 23],
        'Outcome':            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                           0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
}

    df = pd.DataFrame(data)

    cleaned_df = clean_data(df)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_and_split(cleaned_df)
    model = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test)
    threshold_tuning(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()