"""
Bike Rental Demand Prediction - Random Forest Regressor (Hourly Data)

References:
1. This project utilizes logic and methodology inspired by the following Kaggle notebook:
    "Bike Rental Count Prediction using Python" by Lakshmipathi
    URL: https://www.kaggle.com/code/lakshmi25npathi/bike-rental-count-prediction-using-python

2. This project made use of generative AI to assist with code,
    and enhancing the understanding of machine learning concepts.
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error 

# Function to calculate Root Mean Squared Error (RMSE), MAE, and R2 on the ORIGINAL scale
def calculate_metrics(y_true_log, y_pred_log):
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    r2 = r2_score(y_true_exp, y_pred_exp)
    rmse = math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
    mae = mean_absolute_error(y_true_exp, y_pred_exp)
    return r2, rmse, mae

# Function for custom Cross-Validation RMSE (required because CV defaults to 'neg_mean_squared_error')
def custom_rmse_scorer(y_true_log, y_pred_log):
    # Convert back to the original scale
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    # Calculate RMSE
    return math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

# Creates a custom scorer object
rmse_scorer = make_scorer(custom_rmse_scorer, greater_is_better=False) # greater_is_better=False for error metrics

# =========================================================================
# 1) DATA PREPARATION, MODEL TRAINING & CROSS-VALIDATION
# =========================================================================
data = pd.read_csv("hour.csv", parse_dates=['dteday'], index_col='dteday')
data = data.sort_index()
data['cnt'] = np.log1p(data['cnt'])
data['cnt_lag_1'] = data['cnt'].shift(1)
data['hr_sin'] = np.sin(2 * np.pi * data['hr']/24)
data['hr_cos'] = np.cos(2 * np.pi * data['hr']/24)
data.dropna(inplace=True)
data = data.iloc[24:] 

categorical_features = ['season', 'mnth', 'weathersit', 'yr', 'holiday', 'workingday'] 
data_processed = pd.get_dummies(data, columns=categorical_features, drop_first=True)

exclude_cols = ['casual', 'registered', 'cnt', 'instant', 'dteday', 'hr', 'weekday', 'weekday_sin', 'weekday_cos']
features = [col for col in data_processed.columns if col not in exclude_cols]
target = 'cnt'

X = data_processed[features]
y = data_processed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
tscv = TimeSeriesSplit(n_splits=5) 

# Model and parameter selection (RandomForestRegressor) inspired by the referenced Kaggle notebook approach
rf = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# --- CROSS-VALIDATION STEP ---
# R² is standard, but we use negative scores to ensure Cross-Validation operates correctly
cv_r2_scores_log = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)

# Using the custom scorer (rmse_scorer) to obtain RMSE on the original scale
cv_rmse_scores = cross_val_score(rf, X, y, cv=tscv, scoring=rmse_scorer, n_jobs=-1)

cv_mean_r2 = cv_r2_scores_log.mean()
cv_mean_rmse = np.abs(cv_rmse_scores).mean()
# -------------------------------------

# =========================================================================
# 2) CALCULATION OF DETAILED METRICS
# =========================================================================

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

train_r2, train_rmse, train_mae = calculate_metrics(y_train, y_pred_train) 
test_r2, test_rmse, test_mae = calculate_metrics(y_test, y_pred_test)

# Calculate actual (non-log) predictions and values for visualization
y_test_exp = np.expm1(y_test)
y_pred_test_exp = np.expm1(y_pred_test)

# =========================================================================
# 3) FINAL ANALYSIS AND INTERPRETATION 
# =========================================================================

overfitting_gap_r2 = train_r2 - test_r2
overfitting_gap_rmse = test_rmse - train_rmse 

print("\n====================  FINAL ANALYSIS RANDOM FOREST (HOURLY)  ====================")
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Overfitting Gap (R²): {overfitting_gap_r2:+.4f}")
print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f} | Overfitting Gap (RMSE): {overfitting_gap_rmse:+.2f}")
print("---  Cross-Validation (Timeseries - Original Scale) ---")
print(f"Mean CV R² (5 Folds): {cv_mean_r2:.4f}")
print(f"Mean CV RMSE (5 Folds): {cv_mean_rmse:.2f} bikes")
print("===================================================================================\n")

print("---  Overfitting and Generalization ---")
if overfitting_gap_r2 < 0.03 and overfitting_gap_r2 > 0:
    print(f" Excellent Generalization: R² gap is extremely low and positive ({overfitting_gap_r2:+.4f}).")
elif overfitting_gap_r2 <= 0:
    print(f" Negative Gap: This may indicate data leakage or a very small training set. Gap: {overfitting_gap_r2:+.4f}.")
else:
    print(f" Overfitting: The R² gap ({overfitting_gap_r2:+.4f}) is too high. Performance is significantly better on training data.")

print(f"Test RMSE (Bikes): {test_rmse:.2f} | Root Mean Squared Error on the test set.")
print(f"Test MAE (Bikes): {test_mae:.2f} | Mean Absolute Error on the test set.")

# =========================================================================
# 4) VISUALIZATIONS
# =========================================================================

# --- Scatter: Actual vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test_exp, y_pred_test_exp, alpha=0.6, s=45, edgecolors='k', linewidths=0.4)
min_val = min(y_test_exp.min(), y_pred_test_exp.min())
max_val = max(y_test_exp.max(), y_pred_test_exp.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect (y=x)')
plt.title("Actual vs Predicted (Hourly Test Set)", fontsize=16)
plt.xlabel("Actual hourly rentals", fontsize=14)
plt.ylabel("Predicted hourly rentals", fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('hour_rf_scatter_actual_vs_pred.png', dpi=300)
plt.show()

# --- Scatter: Residuals vs Predicted (Test set, original scale) ---
residuals_test = y_test_exp - y_pred_test_exp
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_test_exp, residuals_test, alpha=0.6, s=45, edgecolors='k', linewidths=0.4)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.title("Residuals vs Predicted (Hourly Test Set)", fontsize=16)
plt.xlabel("Predicted hourly rentals", fontsize=14)
plt.ylabel("Residual (actual - predicted)", fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('hour_rf_scatter_residuals_vs_pred.png', dpi=300)
plt.show()

# --- Time Series: Actual vs Predicted over the test period ---
n_test = len(y_test)
test_slice = data.iloc[-n_test:] 
ts_test = test_slice.index + pd.to_timedelta(test_slice['hr'], unit='h')

plt.figure(figsize=(20, 5), dpi=150)
plt.plot(ts_test, np.expm1(y_test), label='Actual', color='steelblue', lw=2)
plt.plot(ts_test, np.expm1(y_pred_test), label='Predicted', color='darkorange', lw=2, linestyle='--')
plt.title('Time Series: Actual vs Predicted (Random Forest, Hourly Test)', fontsize=16)
plt.xlabel('Timestamp (date + hour)', fontsize=14)
plt.ylabel('Bike rentals (count)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('hour_rf_timeseries_actual_vs_pred.png', dpi=300)
plt.show()

# --- Feature Importance (MDI) ---
# Analysis of feature importance follows the logic presented in the referenced Kaggle notebook
importances = rf.feature_importances_         
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, max(5, 0.30 * len(feat_imp))))
feat_imp.plot(kind='barh', color='teal')
plt.title('Random Forest Feature Importance (Hourly)', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.tight_layout()
plt.savefig('hour_rf_feature_importance.png', dpi=300)
plt.show()


print("\n==================== FINAL VERDICT ====================")
print(f"Model: Random Forest Regressor")
print(f"Data: Hour.csv (with Lag & Cyclic Encoding)")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"Test RMSE: {test_rmse:.0f} bikes")
print(f"Test MAE: {test_mae:.0f} bikes") 
print("=======================================================")