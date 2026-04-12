import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Task 1. Create the dataset below using pandas:
data = {
    "cgpa":         [6.5, 7.8, 5.2, 8.9, 6.0, 9.1, 7.2, 5.8, 8.5, 7.0],
    "iq":           [110, 125, 95, 140, 105, 145, 120, 100, 135, 115],
    "grade":        ["B", "A", "C", "A", "B", "A", "B", "C", "A", "B"],
    "placement":    ["yes", "yes", "no", "yes", "no", "yes", "yes", "no", "yes", "yes"]
}
df = pd.DataFrame(data)

# Task 2: Ordinal-encode grade using the custom order C < B < A.
grade_mapping = {"C": 0, "B": 1, "A": 2}
df["grade"] = df["grade"].map(grade_mapping)

# Task 3: Label-encode placement where no -> 0 and yes -> 1.
placement_mapping = {"no": 0, "yes": 1}
df["placement"] = df["placement"].map(placement_mapping)

# Task 4: Split data into features X and target y.
X = df[["cgpa", "iq", "grade"]]
y = df["placement"]

# Task 5: Create train/test split with 90-10 ratio and fixed random_state.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# Task 6: Fit StandardScaler on train data and transform both train and test sets.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 7: Train LogisticRegression and evaluate test accuracy.
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
