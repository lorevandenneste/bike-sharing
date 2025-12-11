import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance # NIEUW

# Functie om de Root Mean Squared Error (RMSE) te berekenen op de OORSPRONKELIJKE schaal
def calculate_metrics(y_true_log, y_pred_log):
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    r2 = r2_score(y_true_exp, y_pred_exp)
    rmse = math.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
    return r2, rmse

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

train_r2, train_rmse = calculate_metrics(y_train, rf.predict(X_train))
test_r2, test_rmse = calculate_metrics(y_test, rf.predict(X_test))

# Cross-Validation scores (gedetailleerd)
cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)
cv_mean_r2 = cv_scores.mean()
cv_std = cv_scores.std()

# Permutation Importance (Op de Testset)
perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='r2')
perm_imp_series = pd.Series(perm_importance.importances_mean, index=X_test.columns).sort_values(ascending=False).head(5)

# Feature Importances (MDI)
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False).head(5)

# =========================================================================
# 3) NIEUWE TERMINAL OUTPUT EN PLOTS
# =========================================================================
print("\n==================== 📊 EINDANALYSE RANDOM FOREST (UURDATA) 📊 ====================")
print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f} | Overfitting Gap: {train_r2 - test_r2:+.4f}")
print("===================================================================================\n")

print("--- 🔬 Overfitting en Generalisatie ---")
if (train_r2 - test_r2) < 0.03:
    print(f"✅ Generalisatie Uitmuntend: Gap van {train_r2 - test_r2:+.4f} is extreem laag.")
else:
    print(f"❌ Overfitting: Gap is hoog.")
print(f"Test RMSE (Fietsen): {test_rmse:.2f}")

print("\n--- ⏳ Time Series Cross-Validation (TSC) ---")
print(f"TSC Mean R²: {cv_mean_r2:.4f} | Standaardafwijking: {cv_std:.4f}")
print("TSC R² per Fold (bewijst robuustheid):")
for i, score in enumerate(cv_scores):
    print(f"  Fold {i+1}: {score:.4f}")
if cv_mean_r2 > 0.9:
    print("✅ TSC Uitmuntend: De validatie is extreem stabiel over de tijd.")

print("\n--- 🔑 Feature Importances (MDI - Op Training) ---")
print(feat_imp.to_string())

print("\n--- 🔑 Permutation Importance (Op Testset - Eerlijker) ---")
print(perm_imp_series.to_string())

# Plot Residuals en Importances opnieuw om de nieuwe data te visualiseren
residuals = np.expm1(y_test) - np.expm1(rf.predict(X_test))
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# A) Residuals Plot
axes[0].scatter(np.expm1(y_test), residuals, alpha=0.5, s=15, color='orange')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_title('Residuals vs Werkelijke Waarde (Idealiter geen patroon)')
axes[0].set_xlabel('Werkelijke Waarde (Fietsen)')
axes[0].set_ylabel('Residual (Fout)')
axes[0].grid(True, linestyle='--', alpha=0.6)
# 

# B) Permutation Importance Plot
axes[1].barh(perm_imp_series.index, perm_imp_series.values, color='purple')
axes[1].set_title('Top-5 Permutation Importance (Op Testset R²)')
axes[1].set_xlabel('Daling in R² bij Shuffling')
axes[1].invert_yaxis()
axes[1].grid(True, linestyle='--', axis='x', alpha=0.6)

plt.tight_layout()
plt.show()

print("\n==================== FINAL VERDICT ====================")
print(f"Model: Random Forest Regressor")
print(f"Data: Hour.csv (met Lag & Cyclische Encoding)")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"RMSE: {test_rmse:.0f} fietsen")
print("=======================================================")