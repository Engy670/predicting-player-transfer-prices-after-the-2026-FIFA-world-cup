# ⚽ AI-Powered Football Player Valuation: Post-2026 World Cup Predictions

A Machine Learning project leveraging **Linear Regression** to predict and analyze football players' transfer market values based on their performance metrics in the **2026 FIFA World Cup** dataset.

---

## 📊 Project Workflow & Pipeline

1. **Dataset Download**: Automatically fetches the latest FIFA World Cup 2026 Player Performance dataset using `kagglehub`.
2. **Data Preprocessing**:
   - Drops identification & categorical non-numeric features (e.g., `player_name`, `team`, `club_name`).
   - Imputes missing numerical values using the **Median** and categorical values using the **Mode**.
   - Applies **One-Hot Encoding** (`pd.get_dummies`) to remaining categorical columns.
3. **Model Training**:
   - Model: `LinearRegression` from `scikit-learn`.
   - Data Split: 80% Training, 20% Testing.
4. **Evaluation**:
   - Evaluates performance using **MAE**, **MSE**, and **R² Score**.
   - Visualizes `Actual vs Predicted` values and extracts feature coefficients.

---

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run script: `python main.py`
