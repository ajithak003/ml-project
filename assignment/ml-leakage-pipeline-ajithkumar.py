from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

#Task 1 — Reproduce and Identify Leakage 
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

scale = StandardScaler()
X_scaled = scale.fit_transform(X)   

x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)    

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)  
print(model.score(x_test, y_test))


#Task 2 — Fix the Workflow Using a Pipeline 

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
print("Cross-validation mean scores:", scores.mean().round(2))
print("Cross-validation std scores:", scores.std().round(2))


#Task 3 — Experiment with Decision Tree Depth 

for depth in [1, 5, 20]:
    
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(x_train, y_train)

    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print(f"Depth: {depth}, Train score: {train_score:.2f},Test score: {test_score:.2f}")


"""
1. Depth: 1, Train score: 0.88,Test score: 0.84 : The model is underfitting, 
as it is too simple to capture the underlying patterns in the data. 
Both train and test scores are relatively low.

2. Depth: 5, Train score: 0.95,Test score: 0.85 : The model is performing well, with a good balance between bias 
and variance. The train score is high, indicating that the model is fitting the training data well, while the 
test score is also good, suggesting that the model generalizes well to unseen data.

3. Depth: 20, Train score: 1.00,Test score: 0.84 : The model is overfitting, as it is too complex and 
captures noise in the training data. The train score is perfect, indicating that the model is fitting 
the training data perfectly, but the test score is lower than the train score, 
suggesting that the model does not generalize well to unseen data. 

In summary, a depth of 5 provides the best balance between underfitting and overfitting, 
while depths of 1 and 20 lead to underfitting and overfitting, respectively."""
