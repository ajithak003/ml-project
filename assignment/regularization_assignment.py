import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV

# Dataset Setup — Do not modify this block
X, y = make_regression(n_samples=200, n_features=20, noise=15, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Task 1: Fit a Ridge model with alpha=10 and print its R² score on test data
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
print("Ridge R² Score:", ridge.score(X_test, y_test))


# Task 2: Fit a Lasso model with alpha=0.5 and print its R² score on test data
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)
print("Lasso R² Score:", lasso.score(X_test, y_test))


# Task 3: Use RidgeCV with alphas=[0.1, 1, 10, 100], print the best alpha and R² score
ridge_cv = RidgeCV(alphas=[0.1, 1, 10, 100])
ridge_cv.fit(X_train, y_train)
print("Best Alpha:", ridge_cv.alpha_)
print("RidgeCV R² Score:", ridge_cv.score(X_test, y_test))
