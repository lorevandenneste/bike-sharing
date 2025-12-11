import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# 1) Dataset laden en indexeren
data = pd.read_csv("day.csv", parse_dates=['dteday'], index_col='dteday')
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

# 4) Rolling walk‑forward functie
def rolling_walk_forward(model, X, y, window_size=365, horizon=1, step_size=7):
    preds, actuals, dates = [], [], []
    n = len(X)
    for t in range(window_size, n - horizon + 1, step_size):
        X_train = X.iloc[t-window_size:t]
        y_train = y.iloc[t-window_size:t]
        X_test  = X.iloc[t:t+horizon]
        y_test  = y.iloc[t:t+horizon]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        preds.extend(y_pred.tolist())
        actuals.extend(y_test.values.tolist())
        dates.extend(X_test.index.tolist())

        if t % 100 == 0:
            print(f"Step {t}/{n}")
    return preds, actuals, dates

# 5) Model instellen
rf = RandomForestRegressor(
    n_estimators=150,   # iets meer bomen
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=1
)

# 6) Rolling walk‑forward uitvoeren
preds, actuals, dates = rolling_walk_forward(rf, X, y, window_size=365, horizon=1, step_size=7)

# 7) Evaluatie
wf_r2  = r2_score(actuals, preds)
wf_rmse = math.sqrt(mean_squared_error(actuals, preds))

print("\n==================== ROLLING WALK-FORWARD PERFORMANCE ====================")
print(f"Observations: {len(actuals)}")
print(f"Rolling WF R²:  {wf_r2:.4f}")
print(f"Rolling WF RMSE:{wf_rmse:.2f}")
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

# 10) Visualisaties
print("\n📊 Generating visuals...")

# A) Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(dates, actuals, label="Actual", color="steelblue")
plt.plot(dates, preds,   label="Predicted", color="darkorange")
plt.title("Rolling Walk-forward: Actual vs Predicted (met seizoensfeatures)")
plt.xlabel("Date"); plt.ylabel("Bike rentals")
plt.legend()
plt.tight_layout()
plt.savefig("rolling_walk_forward_rf_seasonal.png", dpi=300)
plt.show()

# B) Residuals over tijd
residuals = np.array(actuals) - np.array(preds)
plt.figure(figsize=(12,4))
plt.plot(dates, residuals, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.title("Residuals over time (met seizoensfeatures)")
plt.xlabel("Date"); plt.ylabel("Residual")
plt.tight_layout()
plt.savefig("rolling_walk_forward_rf_residuals_seasonal.png", dpi=300)
plt.show()

# C) Residual histogram
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=30, kde=True, color="steelblue")
plt.title("Residual distribution (met seizoensfeatures)")
plt.xlabel("Residual"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("rolling_walk_forward_rf_residual_hist_seasonal.png", dpi=300)
plt.show()

# D) Feature importances
plt.figure(figsize=(8,6))
plt.barh(feat_imp.index, feat_imp.values, color="teal")
plt.title("Feature importances (RF, last fit, met seizoensfeatures)")
plt.tight_layout()
plt.savefig("rolling_walk_forward_rf_feature_importances_seasonal.png", dpi=300)
plt.show()

# 11) Final verdict
print("\n==================== FINAL VERDICT ====================")
print(f"Rolling WF R²: {wf_r2:.4f} ({wf_r2*100:.1f}%)")
print(f"Rolling WF RMSE: {wf_rmse:.0f} bikes")
print(verdict)
print("========================================================\n")
