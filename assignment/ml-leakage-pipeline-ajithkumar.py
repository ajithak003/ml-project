import random
from typing import Dict, List
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def create_synthetic_housing_data(n_records: int = 60, seed: int = 42):
    """
    Create a synthetic housing dataset with required features and target:
    area_sqft, num_bedrooms, age_years, price_lakhs.
    """
    rng = random.Random(seed)

    rows: List[Dict[str, float]] = []

    for _ in range(n_records):
        area_sqft = rng.randint(500, 3500)
        num_bedrooms = rng.randint(1, 5)
        age_years = rng.randint(0, 40)

        base_price = 20
        area_effect = area_sqft * 0.03
        bedroom_effect = num_bedrooms * 4.5
        age_effect = age_years * -0.35
        noise = rng.gauss(0, 5)

        price_lakhs = round(base_price + area_effect + bedroom_effect + age_effect + noise, 2)
        price_lakhs = max(price_lakhs, 8)

        rows.append(
            {
                "area_sqft": area_sqft,
                "num_bedrooms": num_bedrooms,
                "age_years": age_years,
                "price_lakhs": price_lakhs,
            }
        )

    return pd.DataFrame(rows)

def train_model(data: pd.DataFrame):
    """
    Train a linear regression model to predict price_lakhs based on the features.
    """
    X = data[["area_sqft", "num_bedrooms", "age_years"]]
    y = data["price_lakhs"]

    model = LinearRegression()
    model.fit(X, y)
    print("Coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.2f}")
    print(f"  Intercept: {model.intercept_:.2f}")

    X = data[["area_sqft", "num_bedrooms", "age_years"]]
    y_actual = data["price_lakhs"]
    y_pred = model.predict(X)

    comparison_df = pd.DataFrame(
        {"actual_price_lakhs": y_actual, "predicted_price_lakhs": y_pred}
    )
    comparison_df["predicted_price_lakhs"] = comparison_df["predicted_price_lakhs"].round(2)

    print("\nFirst 5 Actual vs Predicted:")
    print(comparison_df.head(5))

    find_mae_rmse_r_squared(y_actual, y_pred)
    plot_residual_histogram(y_actual, y_pred)


def find_mae_rmse_r_squared(y_actual, y_pred):  
    mae = np.mean(np.abs(y_actual - y_pred))  
    rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))  
    r_squared = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))  
    print(f"\nModel Evaluation Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared: {r_squared:.4f}")

    """Model Evaluation Metrics:
    Mean Absolute Error (MAE): 3.33
    Root Mean Squared Error (RMSE): 4.05
    R-squared: 0.9797"""
    # Model performance comments:
    # MAE = 3.33 means predictions are off by about 3.33 lakhs on average, which is low.
    # RMSE = 4.05 is slightly higher than MAE, suggesting a few larger errors exist, but not severe.
    # R-squared = 0.9797 means the model explains about 97.97% of price variation, indicating an excellent fit.
    # Overall, the model is performing very well on this synthetic dataset.




def plot_residual_histogram(y_actual, y_pred):
    """
    Compute residuals and plot their histogram.
    """
    residuals = y_actual - y_pred

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=12, edgecolor="black", alpha=0.75)
    plt.title("Histogram of Residuals")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("residuals_histogram.png")
    plt.close()

    # Residual = actual value - predicted value.
    # A histogram centered near zero and roughly bell-shaped suggests the model's
    # errors are mostly random and unbiased. Strong skew or multiple peaks may
    # indicate missing patterns, outliers, or model misspecification.

if __name__ == "__main__":
    data = create_synthetic_housing_data(n_records=60, seed=42)
    print(data.head(10))
    print(f"\nTotal records: {len(data)}")

    train_model(data)
    

    
    
