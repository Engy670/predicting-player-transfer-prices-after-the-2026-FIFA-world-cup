import kagglehub

path = kagglehub.dataset_download(
    "rauffauzanrambe/fifa-world-cup-2026-player-performance-dataset"
)
print("Path to dataset files:", path)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = None
for f in os.listdir(path):
    if f.endswith(".csv"):
        csv_file = os.path.join(path, f)
        break

print("CSV file found:", csv_file)

df = pd.read_csv(csv_file)

print("\n===== 1) Dataset Introduction =====")
print("Source: Kaggle - FIFA World Cup 2026 Player Performance Dataset")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])
print("\nColumn names:\n", df.columns.tolist())

TARGET = "market_value_eur"

print("\n===== 3) Data Exploration =====")
print("\n--- head() ---")
print(df.head())

print("\n--- info() ---")
print(df.info())

print("\n--- describe() ---")
print(df.describe())

print("\n--- Missing values per column ---")
print(df.isnull().sum())

print("\n===== 4) Preprocessing =====")

columns_to_drop = [
    "player_id", "player_name", "match_id", "match_date",
    "stadium", "city", "team", "club_name", "nationality",
    "opponent", "tournament", "match_result"
]
columns_to_drop = [c for c in columns_to_drop if c in df.columns]
df_model = df.drop(columns=columns_to_drop)

for col in df_model.columns:
    if df_model[col].dtype in [np.float64, np.int64]:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    else:
        df_model[col] = df_model[col].fillna(df_model[col].mode()[0])

categorical_cols = df_model.select_dtypes(include=["object"]).columns.tolist()
if TARGET in categorical_cols:
    categorical_cols.remove(TARGET)

df_model = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

print("Shape after preprocessing:", df_model.shape)

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully.")

y_pred = model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== 9) Evaluation Metrics =====")
print(f"MAE  (Mean Absolute Error): {mae:,.2f}")
print(f"MSE  (Mean Squared Error) : {mse:,.2f}")
print(f"R^2 Score                 : {r2:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red", linestyle="--"
)
plt.xlabel("Actual market_value")
plt.ylabel("Predicted market_value")
plt.title("Actual vs Predicted - Linear Regression")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n===== Top 10 Positive Influencing Features =====")
print(coefficients.head(10))

print("\n===== Top 10 Negative Influencing Features =====")
print(coefficients.tail(10))

print("\n===== Explanation of Results =====")
print("MAE: average absolute difference between actual and predicted market value.")
print("MSE: similar to MAE but squares the difference, penalizing large errors more.")
print("R2 Score: proportion of variance in market value explained by the model.")
print("Linear Regression assumes a linear relationship between the variables.")