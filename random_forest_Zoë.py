# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("bike-sharing/day.csv")


# Select features en target
features = ['temp', 'atemp', 'hum', 'windspeed', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
X = data[features]
y = data['cnt']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Predictions
y_pred = rf.predict(X_test)


# Evaluation

print("R2 Score:", r2_score(y_test, y_pred))

import math
print("RMSE:", math.sqrt(mean_squared_error(y_test, y_pred)))



