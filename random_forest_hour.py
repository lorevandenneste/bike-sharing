import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from sklearn.inspection import permutation_importance 

# Functie om de Root Mean Squared Error (RMSE), MAE en R2 te berekenen op de OORSPRONKELIJKE schaal
def calculate_metrics(y_true_log, y_pred_log):
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    r2 = r2_score(y_true_exp, y_pred_exp)
    rmse = math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
    mae = mean_absolute_error(y_true_exp, y_pred_exp)
    return r2, rmse, mae

# =========================================================================
# 1) HERGEBRUIK DATA PREPARATIE & MODEL TRAINING
# =========================================================================
data = pd.read_csv("bike-sharing/hour.csv", parse_dates=['dteday'], index_col='dteday')
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

rf = RandomForestRegressor(n_estimators=150, max_depth=20, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# =========================================================================
# 2) GEDETAILLEERDE METRIEKEN BEREKENEN
# =========================================================================

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

train_r2, train_rmse, train_mae = calculate_metrics(y_train, y_pred_train) 
test_r2, test_rmse, test_mae = calculate_metrics(y_test, y_pred_test)

# Bereken de werkelijke (niet-log) voorspellingen en waarden voor plots
y_test_exp = np.expm1(y_test)
y_pred_test_exp = np.expm1(y_pred_test)

# =========================================================================
# 3) TERMINAL OUTPUT (Inclusief alle gevraagde metrieken)
# (Teksten in de terminaloutput worden onveranderd gelaten in het Nederlands)
# =========================================================================

overfitting_gap_r2 = train_r2 - test_r2
overfitting_gap_rmse = test_rmse - train_rmse 

print("\n==================== 📊 EINDANALYSE RANDOM FOREST (UURDATA) 📊 ====================")
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Overfitting Gap (R²): {overfitting_gap_r2:+.4f}")
print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f} | Overfitting Gap (RMSE): {overfitting_gap_rmse:+.2f}")
print("===================================================================================\n")

print("--- 🔬 Overfitting en Generalisatie ---")
if overfitting_gap_r2 < 0.03 and overfitting_gap_r2 > 0:
    print(f"✅ Generalisatie Uitmuntend: Gap van R² is extreem laag en positief ({overfitting_gap_r2:+.4f}).")
elif overfitting_gap_r2 <= 0:
    print(f"⚠️ Gap is Negatief: Dit kan wijzen op data leakage (of een heel kleine trainingsset). Gap: {overfitting_gap_r2:+.4f}.")
else:
    print(f"❌ Overfitting: De R² gap ({overfitting_gap_r2:+.4f}) is te hoog. Model presteert veel beter op training.")

print(f"Test RMSE (Fietsen): {test_rmse:.2f} | Gemiddelde kwadratische fout op de testset.")
print(f"Test MAE (Fietsen): {test_mae:.2f} | Gemiddelde absolute fout op de testset.")

# =========================================================================
# 4) VEREENVOUDIGDE VISUALS: ENKEL ACTUAL VS PREDICTED (IN HET ENGELS)
# =========================================================================

# Maak een enkele figuur
plt.figure(figsize=(9, 7))

# --- Plot: Actual vs Predicted ---
# Scatter plot kleur is gekozen als 'blue' om een duidelijke vergelijking te maken met standaard lineaire regressie plots
plt.scatter(y_test_exp, y_pred_test_exp, alpha=0.5, s=25, edgecolors='k', linewidths=0.5)

# Perfecte voorspellingslijn (y=x)
min_val = y_test_exp.min()
max_val = y_test_exp.max()
# Lijn is rood ('red') en gestippeld, zoals vaak gebruikt
plt.plot([min_val, max_val], [min_val, max_val], 
         'r--', lw=2, label='Perfect Prediction (y=x)')

# Titels en labels in het Engels
plt.title('Actual vs Predicted Plot (Random Forest Hour Data)', fontsize=14)
plt.xlabel('Actual Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

print("\n==================== FINAL VERDICT ====================")
print(f"Model: Random Forest Regressor")
print(f"Data: Hour.csv (met Lag & Cyclische Encoding)")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"Test RMSE: {test_rmse:.0f} fietsen")
print(f"Test MAE: {test_mae:.0f} fietsen") 
print("=======================================================")