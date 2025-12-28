import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# 1) Dataset laden en indexeren
data = pd.read_csv("bike-sharing/day.csv", parse_dates=['dteday'], index_col='dteday')
data = data.sort_index()

# 2) Feature Engineering: lags & rollings
data['cnt_lag_1']  = data['cnt'].shift(1)
data['cnt_lag_7']  = data['cnt'].shift(7)
data['cnt_lag_14'] = data['cnt'].shift(14)
data['cnt_lag_30'] = data['cnt'].shift(30)
data['cnt_roll_7']  = data['cnt'].shift(1).rolling(7, min_periods=7).mean()
data['cnt_roll_30'] = data['cnt'].shift(1).rolling(30, min_periods=30).mean()

# 3) Cyclische seizoensfeatures
data['dayofyear'] = data.index.dayofyear
data['sin_doy'] = np.sin(2*np.pi*data['dayofyear']/365)
data['cos_doy'] = np.cos(2*np.pi*data['dayofyear']/365)

data['month'] = data.index.month
data['sin_month'] = np.sin(2*np.pi*data['month']/12)
data['cos_month'] = np.cos(2*np.pi*data['month']/12)

# Drop NaN’s
data.dropna(inplace=True)

features = ['temp','hum','windspeed','season','yr','mnth','holiday','weekday',
            'workingday','weathersit','cnt_lag_1','cnt_lag_7','cnt_lag_14',
            'cnt_lag_30','cnt_roll_7','cnt_roll_30',
            'sin_doy','cos_doy','sin_month','cos_month']
X = data[features]
y = data['cnt']

# 4) Rolling walk‑forward functie (AANGEPAST)
def rolling_walk_forward(model, X, y, window_size=365, horizon=1, step_size=7):
    preds, actuals, dates = [], [], []
    # NIEUW: MAE per stap tracken
    step_train_mae, step_test_mae = [], [] 
    
    n = len(X)
    for t in range(window_size, n - horizon + 1, step_size):
        X_train = X.iloc[t-window_size:t]
        y_train = y.iloc[t-window_size:t]
        X_test  = X.iloc[t:t+horizon]
        y_test  = y.iloc[t:t+horizon]

        model.fit(X_train, y_train)
        
        # Train voorspellingen en MAE berekenen (voor de gap)
        y_pred_train = model.predict(X_train)
        step_train_mae.append(mean_absolute_error(y_train, y_pred_train))
        
        # Test voorspellingen en MAE berekenen (voor de gap)
        y_pred = model.predict(X_test)
        step_test_mae.append(mean_absolute_error(y_test, y_pred))

        preds.extend(y_pred.tolist())
        actuals.extend(y_test.values.tolist())
        dates.extend(X_test.index.tolist())

        if t % 100 == 0:
            print(f"Step {t}/{n}")
            
    # RETOURNEERT NU OOK DE MAE LIJSTEN
    return preds, actuals, dates, step_train_mae, step_test_mae

# 5) Model instellen
rf = RandomForestRegressor(
    n_estimators=150,   
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=1
)

# 6) Rolling walk‑forward uitvoeren (AANGEPAST OM 5 WAARDEN TE ONTVANGEN)
preds, actuals, dates, step_train_mae, step_test_mae = rolling_walk_forward(rf, X, y, window_size=365, horizon=1, step_size=7)

# 7) Evaluatie
wf_r2  = r2_score(actuals, preds)
wf_rmse = math.sqrt(mean_squared_error(actuals, preds))
wf_mae  = mean_absolute_error(actuals, preds)

# NIEUW: Overfitting-gap (gemiddeld over stappen)
# Gap = Test MAE - Train MAE. Positief getal betekent overfitting (testfout is hoger)
avg_mae_gap = np.mean(step_test_mae) - np.mean(step_train_mae)


print("\n==================== ROLLING WALK-FORWARD PERFORMANCE ====================")
print(f"Observations: {len(actuals)}")
print(f"Rolling WF R²:  {wf_r2:.4f}")
print(f"Rolling WF RMSE:{wf_rmse:.2f}")
print(f"Rolling WF MAE: {wf_mae:.2f}")

# Overfitting Gap Print
print(f"\n--- Overfitting Gap (Avg Test MAE - Avg Train MAE) ---")
print(f"Avg MAE Gap: +{avg_mae_gap:.2f} fietsen")
if avg_mae_gap > 100:
    print("❌ Hoge Gap: De fout op de testset is significant hoger dan op de training.")
elif avg_mae_gap > 30:
    print("⚠️ Matige Gap: Het model presteert nog steeds significant beter op training dan op de test.")
else:
    print("✅ Lage Gap: Goede generalisatie. Het model is robuust.")
print("===========================================================\n")

# 8) Interpretatie
print("📌 INTERPRETATION:")
if wf_r2 >= 0.80:
    verdict = "✅ Goede generalisatie: Rolling window score is hoog."
elif wf_r2 >= 0.70:
    verdict = "✓ Acceptabel: Model bruikbaar, maar tuning aanbevolen."
else:
    verdict = "⚠️ Matig: Overweeg meer features of andere modellen."
print(verdict)

# 9) Feature importance (laatste fit)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)
print("\nTop-5 belangrijkste features:")
print(feat_imp.head(5))

# ===================== VISUALS =====================
print("\n📊 Generating visualization...")


# 1) Actual vs Predicted (scatter) met perfecte lijn
plt.figure(figsize=(8, 6))
plt.scatter(actuals, preds, alpha=0.6, s=40, edgecolors='k', linewidths=0.4)
min_val = min(min(actuals), min(preds))
max_val = max(max(actuals), max(preds))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect (y=x)')
plt.title('Actual vs Predicted (Rolling Walk-forward, Daily)', fontsize=16)
plt.xlabel('Actual daily rentals', fontsize=14)
plt.ylabel('Predicted daily rentals', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('day_rf_wf_scatter_actual_vs_pred.png', dpi=300)
plt.show()

# 2) Residual plot (residuals vs predicted)
residuals_wf = np.array(actuals) - np.array(preds)
plt.figure(figsize=(8, 6))
plt.scatter(preds, residuals_wf, alpha=0.6, s=40, edgecolors='k', linewidths=0.4)
plt.axhline(0, color='r', linestyle='--', lw=2)
plt.title('Residuals vs Predicted (Rolling Walk-forward, Daily)', fontsize=16)
plt.xlabel('Predicted daily rentals', fontsize=14)
plt.ylabel('Residuals (actual - predicted)', fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('day_rf_wf_residuals_vs_pred.png', dpi=300)
plt.show()

# 3) Time series: Actual vs Predicted
plt.figure(figsize=(12, 5))
plt.plot(dates, actuals, label='Actual', color='steelblue', lw=2)
plt.plot(dates, preds,  label='Predicted', color='darkorange', lw=2, linestyle='--')
plt.title('Time Series: Actual vs Predicted (Rolling Walk-forward, Daily)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Bike rentals (count)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('day_rf_wf_timeseries_actual_vs_pred.png', dpi=300)
plt.show()

# 4) Residuals over time
plt.figure(figsize=(12, 4))
plt.plot(dates, residuals_wf, color='purple', lw=1.8)
plt.axhline(0, color='gray', linestyle='--', lw=1.5)
plt.title('Residuals over Time (Rolling Walk-forward, Daily)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('day_rf_wf_residuals_over_time.png', dpi=300)
plt.show()

# 5) Feature importance (bar chart)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, max(5, 0.3 * len(feat_imp))))
feat_imp.plot(kind='barh', color='teal')
plt.title('Random Forest Feature Importance (Daily)', fontsize=16)
plt.xlabel('Importance', fontsize=14)
plt.tight_layout()
plt.savefig('day_rf_feature_importance.png', dpi=300)

plt.show()


# 10) Final verdict
print("\n==================== FINAL VERDICT ====================")
print(f"Rolling WF R²: {wf_r2:.4f} ({wf_r2*100:.1f}%)")
print(f"Rolling WF RMSE: {wf_rmse:.0f} bikes")
print(f"Rolling WF MAE:  {wf_mae:.2f} bikes")
# Toevoegen van de MAE gap aan het Final Verdict
print(f"Avg MAE Overfitting Gap: +{avg_mae_gap:.2f} bikes")
print(verdict)
print("========================================================\n")