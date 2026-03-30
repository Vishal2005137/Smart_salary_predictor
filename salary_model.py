# ==========================================
# Advanced HR Salary Prediction Model
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("Chhattisgarh_job_Market_Dataset.csv")

print("Dataset Shape:", df.shape)


# ==========================================
# 2. Remove ID Column
# ==========================================

# df = df.drop(columns=["Employee_ID"])


# ==========================================
# 3. Separate Features & Target
# ==========================================

y = df["Salary"]
X = df.drop("Salary", axis=1)


# ==========================================
# 4. Log Transform Target
# ==========================================

y = np.log1p(y)


# ==========================================
# 5. Identify Categorical & Numerical Columns
# ==========================================

categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns


# ==========================================
# 6. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)


# ==========================================
# 7. XGBoost Model
# ==========================================

model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=2,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.3,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 8. Full Pipeline
# ==========================================

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==========================================
# 9. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==========================================
# 10. Train Model
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 11. Predictions
# ==========================================

y_pred_log = pipeline.predict(X_test)

# Convert back from log
y_test_original = np.expm1(y_test)
y_pred_original = np.expm1(y_pred_log)


# ==========================================
# 12. Evaluation
# ==========================================

print("\nMODEL PERFORMANCE")
print("MAE:", mean_absolute_error(y_test_original, y_pred_original))
print("RMSE:", np.sqrt(mean_squared_error(y_test_original, y_pred_original)))
print("R2 Score:", r2_score(y_test_original, y_pred_original))


# ==========================================
# 13. Save Model
# ==========================================

joblib.dump(pipeline, "hr_salary_prediction_model.pkl")

print("\nModel Saved Successfully!")

from sklearn.model_selection import cross_val_score

# scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
# print("Cross Validation R2 Scores:", scores)
# print("Average R2:", scores.mean())