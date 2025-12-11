import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

# Functie om de Root Mean Squared Error (RMSE) te berekenen op de OORSPRONKELIJKE schaal
def neg_rmse_exp(y_true, y_pred):
    """Berekent de RMSE na het toepassen van de inverse log transformatie."""
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    return -np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

# =========================================================================
# 1) Laden van de dataset en Datum Indexering
# =========================================================================
try:
    # >>> GEBRUIK HIER hour.csv <<<
    data = pd.read_csv("bike-sharing/hour.csv", parse_dates=['dteday'], index_col='dteday')
except FileNotFoundError:
    print("Fout: hour.csv niet gevonden. Controleer het bestandspad.")
    exit()

data = data.sort_index()
print(f"📌 Dataset geladen: {len(data)} uurlijkse rijen.")

# =========================================================================
# 2) Feature Engineering & Log Transformatie
# =========================================================================

# Log-Transformatie van de target
data['cnt'] = np.log1p(data['cnt'])

# Lag-Features (Vorige uur)
data['cnt_lag_1'] = data['cnt'].shift(1)

# >>> NIEUW: CYCLISCHE ENCODING voor UUR (hr) en DAG (weekday) <<<
# Dit helpt het model te begrijpen dat 23 uur dicht bij 0 uur ligt.
data['hr_sin'] = np.sin(2 * np.pi * data['hr']/24)
data['hr_cos'] = np.cos(2 * np.pi * data['hr']/24)
data['weekday_sin'] = np.sin(2 * np.pi * data['weekday']/7)
data['weekday_cos'] = np.cos(2 * np.pi * data['weekday']/7)


# Verwijder NaN-waarden
data.dropna(inplace=True)
data = data.iloc[24:] # Verwijder de eerste 24 uur om stabiliteit te garanderen
print(f"📌 Data voor training: {len(data)} rijen na stabilisatie.")


# =========================================================================
# 3) Categorische Features verwerken (One-Hot Encoding)
# =========================================================================
# Gebruik hier alleen 'season', 'mnth', 'weathersit', 'yr', 'holiday', 'workingday'
categorical_features = ['season', 'mnth', 'weathersit', 'yr', 'holiday', 'workingday'] 
data_processed = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Definitieve features
exclude_cols = ['casual', 'registered', 'cnt', 'instant', 'dteday', 'hr', 'weekday']
features = [col for col in data_processed.columns if col not in exclude_cols]

target = 'cnt'

X = data_processed[features]
y = data_processed[target]

# Splitsen van de data (shuffle=False - Eerlijke Time Series Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# TimeSeriesSplit voor CV (geen min_train_size nodig door de grote dataset)
tscv = TimeSeriesSplit(n_splits=5) 

# =========================================================================
# 4) MODEL TUNING & TRAINING: XGBoost
# =========================================================================

print("--- 🧠 Start XGBoost Training op UURDATA ---")

# Gebruik de succesvolle parameters van de dag-dataset
best_params = {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.7}

rf = XGBRegressor(
    random_state=42, 
    n_jobs=-1, 
    objective='reg:squarederror', 
    tree_method='hist',
    **best_params
)

rf.fit(X_train, y_train)

# =========================================================================
# 5) Evalueren (MET INVERSE TRANSFORMATIE)
# =========================================================================

# Voorspellingen
y_test_pred_log = rf.predict(X_test)

# **INVERSE TRANSFORMATIE VOOR METRIEKEN**
y_test_pred = np.expm1(y_test_pred_log)
y_test_exp = np.expm1(y_test)

# Metriek berekening (op de OORSPRONKELIJKE schaal)
test_r2 = r2_score(y_test_exp, y_test_pred)
test_rmse = math.sqrt(mean_squared_error(y_test_exp, y_test_pred))

# Cross-Validation scores (op de gehele dataset)
cv_scores = cross_val_score(rf, X, y, cv=tscv, scoring='r2', n_jobs=-1)
cv_mean_r2 = cv_scores.mean()

# =========================================================================
# 6) Resultaten Printen
# =========================================================================

print("\n==================== MODEL PERFORMANCE (XGBoost UURDATA) ====================")
print("⚠️ Validatie is EERLIJK (Shuffle=False). Hoge R² wordt verwacht.")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE:{test_rmse:.2f} fietsen")
print(f"TSC CV Mean R²: {cv_mean_r2:.4f}")
print("=============================================================================\n")

print("==================== FINAL VERDICT ====================")
print(f"Test R²: {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"RMSE: {test_rmse:.0f} fietsen")
print("=======================================================")