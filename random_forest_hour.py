import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error 
from sklearn.inspection import permutation_importance 

# Functie om de Root Mean Squared Error (RMSE), MAE en R2 te berekenen op de OORSPRONKELIJKE schaal
def calculate_metrics(y_true_log, y_pred_log):
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    r2 = r2_score(y_true_exp, y_pred_exp)
    rmse = math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
    mae = mean_absolute_error(y_true_exp, y_pred_exp)
    return r2, rmse, mae

# Functie voor custom Cross-Validation RMSE (nodig omdat CV standaard 'neg_mean_squared_error' gebruikt)
def custom_rmse_scorer(y_true_log, y_pred_log):
    # Converteer terug naar de oorspronkelijke schaal
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    # Bereken de RMSE
    return math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

# Maak een custom scorer object
rmse_scorer = make_scorer(custom_rmse_scorer, greater_is_better=False) # greater_is_better=False voor foutmetingen

# =========================================================================
# 1) HERGEBRUIK DATA PREPARATIE & MODEL TRAINING + CROSS-VALIDATION
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

# --- NIEUWE CROSS-VALIDATION STAP ---
# R² is standaard, maar we moeten de scores Negatief maken om Cross-Validation correct te laten werken
cv_r2_scores_log = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)

# Aangezien we een custom scorer (rmse_scorer) hebben gedefinieerd die de log-terugtransformatie doet, 
# kunnen we deze direct gebruiken om de RMSE op de oorspronkelijke schaal te krijgen.
cv_rmse_scores = cross_val_score(rf, X, y, cv=tscv, scoring=rmse_scorer, n_jobs=-1)

cv_mean_r2 = cv_r2_scores_log.mean()
cv_mean_rmse = np.abs(cv_rmse_scores).mean()
# -------------------------------------

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
# 3) FINAL ANALYSIS EN INTERPRETATION 
# =========================================================================

overfitting_gap_r2 = train_r2 - test_r2
overfitting_gap_rmse = test_rmse - train_rmse 

print("\n==================== 📊 EINDANALYSE RANDOM FOREST (UURDATA) 📊 ====================")
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Overfitting Gap (R²): {overfitting_gap_r2:+.4f}")
print(f"Train RMSE: {train_rmse:.2f} | Test RMSE: {test_rmse:.2f} | Overfitting Gap (RMSE): {overfitting_gap_rmse:+.2f}")
print("--- 🔄 Cross-Validation (Timeseries - Oorspronkelijke Schaal) ---")
print(f"Gemiddelde CV R² (5 Folds): {cv_mean_r2:.4f}")
print(f"Gemiddelde CV RMSE (5 Folds): {cv_mean_rmse:.2f} fietsen")
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
# 4) VISUALS
# =========================================================================

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

# --- 2) Scatter: Residuals vs Predicted (testset, originele schaal) ---
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

# --- 3) Time Series: Actual vs Predicted over de testperiode ---
# We reconstrueren een unieke tijd-as (datum + uur) voor de testset.
# Tip: train_test_split met shuffle=False splitst de laatste 20% als test; we gebruiken dezelfde lengte om het origineel te snijden.
n_test = len(y_test)
test_slice = data.iloc[-n_test:]  # 'data' is je originele hourly dataframe met index=dteday en kolom 'hr'
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

# --- 4) Feature importance (MDI) ---
importances = rf.feature_importances_         # RandomForestRegressor attribute
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
print(f"Data: Hour.csv (met Lag & Cyclische Encoding)")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"Test RMSE: {test_rmse:.0f} fietsen")
print(f"Test MAE: {test_mae:.0f} fietsen") 
print("=======================================================")