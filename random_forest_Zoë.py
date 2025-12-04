import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# 1) Load dataset en Datum Indexering 🗓️
# Gebruik 'dteday' als datum en index
data = pd.read_csv("bike-sharing/day.csv", parse_dates=['dteday'], index_col='dteday')
# Sorteren op datum is cruciaal voor Lag-features en shuffle=False
data = data.sort_index()

# 2) Feature Engineering: Lag-Features (Vorige Dag/Week)
# Voeg het aantal verhuurde fietsen van gisteren (t-1) en vorige week (t-7) toe
data['cnt_lag_1'] = data['cnt'].shift(1)
data['cnt_lag_7'] = data['cnt'].shift(7)

# Verwijder de rijen met NaN-waarden die zijn ontstaan door de shift (de eerste 7 dagen)
data.dropna(inplace=True)

# 3) Features & target bijwerken
features = ['temp', 'hum', 'windspeed',
            'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit',
            'cnt_lag_1', 'cnt_lag_7'] # NIEUWE FEATURES TOEGEVOEGD
X = data[features]
y = data['cnt']

# 4) Train-test split (shuffle=False is behouden, nu op de gezuiverde data)
# Let op: test_size is nu 20% van de resterende data na dropna
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 5) Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
# U kunt hier nu ook de gesuggereerde hyperparameter tuning toevoegen, bv. max_depth=10
# rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42) 
rf.fit(X_train, y_train)

# 6) Cross-validation
# TimeSeriesSplit is correct en behouden
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='r2')
print("Cross-validation R² scores:", cv_scores)
print("Gemiddelde CV R²:", cv_scores.mean())

# 7) Predictions
y_pred = rf.predict(X_test)

# 8) Metrics
train_r2 = r2_score(y_train, rf.predict(X_train))
test_r2 = r2_score(y_test, y_pred)
train_rmse = math.sqrt(mean_squared_error(y_train, rf.predict(X_train)))
test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
gap_r2 = train_r2 - test_r2
gap_rmse = test_rmse - train_rmse

print("\n==================== MODEL PERFORMANCE ====================")
print(f"Train R²:  {train_r2:.4f} | Test R²:  {test_r2:.4f} | Gap: {gap_r2:+.4f}")
print(f"Train RMSE:{train_rmse:.2f} | Test RMSE:{test_rmse:.2f} | Gap: {gap_rmse:+.2f}")
print(f"CV Mean R²: {cv_scores.mean():.4f} | CV Std: {cv_scores.std():.4f}")
print("===========================================================\n")

# ===================== INTERPRETATION =====================
print("📌 INTERPRETATION:")
if gap_r2 > 0.10 and test_r2 < 0.80:
    verdict = "⚠️ Overfitting detected: Grote gap en matige testscore."
elif gap_r2 > 0.05 and test_r2 >= 0.80:
    verdict = "⚠️ Lichte overfitting: Gap is merkbaar, maar testscore blijft hoog."
elif train_r2 < 0.70 and test_r2 < 0.70:
    verdict = "⚠️ Underfitting: Zowel train als test scores zijn laag."
elif abs(gap_r2) < 0.05 and test_r2 >= 0.80:
    verdict = "✅ Goede generalisatie: Kleine gap en hoge scores."
else:
    verdict = "ℹ️ Model is redelijk, tuning kan helpen om bias/variance te verbeteren."
print(verdict)

# Absolute kwaliteit
if test_r2 >= 0.85:
    quality = "🎯 Excellent: Random Forest werkt zeer goed voor deze dataset."
elif test_r2 >= 0.80:
    quality = "👍 Goed: Prestaties zijn sterk, maar er is ruimte voor fine-tuning."
elif test_r2 >= 0.75:
    quality = "✓ Acceptabel: Model is bruikbaar, maar tuning wordt aanbevolen."
else:
    quality = "❌ Matig: Overweeg hyperparameter tuning of andere modellen."
print(quality)
print("===========================================================\n")

# ===================== FEATURE IMPORTANCE =====================
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("Top-5 belangrijkste features:")
print(feat_imp.head(5))

# ===================== PERMUTATION IMPORTANCE =====================
try:
    # Permutation Importance wordt uitgevoerd op de TEST set (X_test)
    perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=42, scoring='r2')
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
    print("\nPermutation Importance (Top 5):")
    print(perm_imp.head(5))
except Exception as e:
    print(f"Permutation importance niet beschikbaar: {e}")

# ===================== VISUALS =====================
print("\n📊 Generating visuals...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# A) Actual vs Predicted
axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=40, edgecolor='k')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_title('Actual vs Predicted')
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')

# B) Residuals vs Predicted
residuals = y_test - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=40, edgecolor='k')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residuals vs Predicted')

# C) Residual Histogram
sns.histplot(residuals, bins=30, kde=True, ax=axes[0, 2], color='steelblue')
axes[0, 2].set_title('Residual Distribution')

# D) Feature Importances
axes[1, 0].barh(feat_imp.index, feat_imp.values, color='teal')
axes[1, 0].set_title('Feature Importances')

# E) Learning Curve
# Gebruik X en y die de lag features bevatten
train_sizes, train_scores, test_scores = learning_curve(
    rf, X, y, cv=tscv, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 8)
)
axes[1, 1].plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train R²')
axes[1, 1].plot(train_sizes, test_scores.mean(axis=1), 'o-', label='CV R²')
axes[1, 1].set_title('Learning Curve')
axes[1, 1].legend()

# F) CV Score Boxplot
sns.boxplot(y=cv_scores, ax=axes[1, 2], color='lightgray')
axes[1, 2].set_title('CV R² Distribution')

plt.tight_layout()
plt.savefig('rf_diagnostics_enhanced.png', dpi=300)
plt.show()

# ===================== FINAL VERDICT =====================
print("\n==================== FINAL VERDICT ====================")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"RMSE: {test_rmse:.0f} bikes")
print(f"Overfitting status: {'Ja, mild' if gap_r2 > 0.05 else 'Nee'} (Gap R² = {gap_r2:.3f})")
print(verdict)
print(quality)
print("========================================================\n")