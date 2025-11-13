import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("day.csv")

# Define features and target
categorical = ['season', 'mnth', 'weekday', 'weathersit', 'holiday', 'workingday', 'yr']
numeric = ['temp', 'atemp', 'hum', 'windspeed']
target = 'cnt'

X = df.drop(columns=[target, 'casual', 'registered', 'dteday', 'instant'])
y = df[target]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical)
    ]
)

# Example: fit_transform the training data
X_processed = preprocessor.fit_transform(X)
print("Shape after preprocessing:", X_processed.shape)