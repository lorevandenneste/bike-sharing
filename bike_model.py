"""
Bike Rental Demand Prediction using Linear Regression
Based on: https://www.kaggle.com/code/sagarpavan123/bike-rentals-prediction-using-linear-regression

Dataset: UCI Bike Sharing Dataset (day.csv)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set visualization style
sns.set_style('whitegrid')

print("="*70)
print("BIKE RENTAL DEMAND PREDICTION - LINEAR REGRESSION")
print("="*70)

# ============================================================================
# STEP 1: DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n📂 STEP 1: Data Loading and Preprocessing")
print("-"*70)

# Load dataset
df = pd.read_csv("day.csv")
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Define features and target
categorical = ['season', 'yr', 'mnth', 'weekday', 'weathersit', 'holiday', 'workingday']
numeric = ['temp', 'atemp', 'hum', 'windspeed']
target = 'cnt'

print(f"\nFeatures:")
print(f"  - Categorical: {categorical}")
print(f"  - Numeric: {numeric}")
print(f"  - Target: {target}")

# Prepare X and y
X = df[categorical + numeric]
y = df[target]

# Split BEFORE preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # shuffle=False preserves time order
)

print(f"\n✓ Train/Test Split:")
print(f"  - Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"  - Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric), # sets all numbers on the same scale
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical) # categories -> binary columns
    ]
)

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\n✓ Preprocessing complete:")
print(f"  - Training shape: {X_train_processed.shape}")
print(f"  - Test shape: {X_test_processed.shape}")
print(f"  - Features after encoding: {X_train_processed.shape[1]}")

# ============================================================================
# STEP 2: MODEL TRAINING - LINEAR REGRESSION
# ============================================================================
print("\n🤖 STEP 2: Model Training")
print("-"*70)

# Create and train Linear Regression model
model = LinearRegression()
print("Training Linear Regression model...")
model.fit(X_train_processed, y_train)
print("✓ Model trained successfully!")

# ============================================================================
# STEP 3: MAKING PREDICTIONS
# ============================================================================
print("\n🔮 STEP 3: Making Predictions")
print("-"*70)

# Predict on both training and test sets
y_train_pred = model.predict(X_train_processed)
y_test_pred = model.predict(X_test_processed)
print("✓ Predictions generated")

# ============================================================================
# STEP 4: MODEL EVALUATION
# ============================================================================
print("\n📊 STEP 4: Model Evaluation")
print("-"*70)

# Calculate metrics for training set
train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

# Calculate metrics for test set
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\n📈 TRAINING SET PERFORMANCE:")
print(f"  R² Score:  {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"  RMSE:      {train_rmse:.2f} bikes")
print(f"  MAE:       {train_mae:.2f} bikes")

print("\n📉 TEST SET PERFORMANCE:")
print(f"  R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"  RMSE:      {test_rmse:.2f} bikes")
print(f"  MAE:       {test_mae:.2f} bikes")

# Overfitting check
print("\n🔍 OVERFITTING CHECK:")
r2_diff = train_r2 - test_r2
print(f"  R² difference (train - test): {r2_diff:.4f}")
if abs(r2_diff) < 0.05:
    print("  ✓ Good! No significant overfitting.")
elif abs(r2_diff) < 0.10:
    print("  ⚠️  Minor overfitting detected.")
else:
    print("  ❌ Significant overfitting detected!")

# ============================================================================
# STEP 5: FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================================
print("\n🔍 STEP 5: Feature Importance Analysis")
print("-"*70)

# Get feature names and coefficients
feature_names = preprocessor.get_feature_names_out()
coefficients = model.coef_

# Create dataframe for better visualization
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n📋 TOP 10 MOST IMPORTANT FEATURES:")
print(f"\n{'Rank':<6} {'Feature':<40} {'Coefficient':<15} {'Impact'}")
print("-"*80)

for i, (idx, row) in enumerate(coef_df.head(10).iterrows(), 1):
    feature = row['Feature'].split('__')[-1]  # Shorten feature name
    coef = row['Coefficient']
    impact = "↑ Increases" if coef > 0 else "↓ Decreases"
    print(f"{i:<6} {feature:<40} {coef:>+14.2f} {impact}")

# ============================================================================
# STEP 6: SAMPLE PREDICTIONS
# ============================================================================
print("\n📋 STEP 6: Sample Predictions")
print("-"*70)

print("\nFirst 10 Test Predictions:")
print(f"\n{'Actual':<12} {'Predicted':<12} {'Difference':<12} {'% Error':<12}")
print("-"*60)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    diff = actual - predicted
    pct_error = (abs(diff) / actual * 100) if actual > 0 else 0
    print(f"{actual:<12.0f} {predicted:<12.0f} {diff:>+12.0f} {pct_error:>11.1f}%")

# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print("\n📊 STEP 7: Creating Visualizations")
print("-"*70)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Bike Rental Demand Prediction - Linear Regression Results', 
             fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Bike Rentals', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Predicted Bike Rentals', fontsize=11, fontweight='bold')
axes[0, 0].set_title(f'Actual vs Predicted (Test Set)\nR² = {test_r2:.4f}', 
                     fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals Plot
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Residual Plot\n(Should be randomly scattered around 0)', 
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Time Series - Actual vs Predicted
test_indices = range(len(y_test))
axes[1, 0].plot(test_indices, y_test.values, 'o-', label='Actual', 
                linewidth=2, markersize=4, alpha=0.7)
axes[1, 0].plot(test_indices, y_test_pred, 'x-', label='Predicted', 
                linewidth=2, markersize=4, alpha=0.7)
axes[1, 0].fill_between(test_indices, y_test.values, y_test_pred, 
                        alpha=0.2, color='red', label='Error')
axes[1, 0].set_xlabel('Test Sample Index (Time →)', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Number of Bike Rentals', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Time Series: Actual vs Predicted', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Top 10 Feature Coefficients
top_10_features = coef_df.head(10).copy()
top_10_features['Short_Name'] = top_10_features['Feature'].apply(lambda x: x.split('__')[-1])
colors = ['green' if c > 0 else 'red' for c in top_10_features['Coefficient']]

axes[1, 1].barh(range(10), top_10_features['Coefficient'].values, 
                color=colors, alpha=0.7, edgecolor='black', linewidth=1)
axes[1, 1].set_yticks(range(10))
axes[1, 1].set_yticklabels(top_10_features['Short_Name'].values, fontsize=9)
axes[1, 1].set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Top 10 Feature Coefficients\n(Green=Positive | Red=Negative)', 
                     fontsize=12, fontweight='bold')
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[1, 1].grid(True, alpha=0.3, axis='x')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.savefig('bike_prediction_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'bike_prediction_results.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"""
📊 Linear Regression Performance:
   R² Score (Test):    {test_r2:.4f} ({test_r2*100:.1f}% variance explained)
   RMSE (Test):        {test_rmse:.0f} bikes
   MAE (Test):         {test_mae:.0f} bikes

💡 Interpretation:
   - The model predicts with an average error of {test_mae:.0f} bikes
   - It explains {test_r2*100:.1f}% of the variance in demand
   
🎯 Model Quality:""")

if test_r2 > 0.85:
    print("   ⭐⭐⭐ Excellent performance!")
elif test_r2 > 0.75:
    print("   ⭐⭐ Good performance!")
elif test_r2 > 0.65:
    print("   ⭐ Acceptable performance")
else:
    print("   ⚠️  Poor performance - consider other models")

print(f"""
📁 Output Files:
   ✓ bike_prediction_results.png (visualizations)

🔄 Next Steps:
   1. Try Random Forest for better performance
   2. Perform hyperparameter tuning
   3. Add cross-validation
   4. Feature engineering (interaction terms)

📚 References:
   - Kaggle: https://www.kaggle.com/code/sagarpavan123/bike-rentals-prediction-using-linear-regression
   - Scikit-learn: https://scikit-learn.org/stable/modules/linear_model.html
   - Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
""")

print("="*70)
print("✅ Analysis Complete!")
print("="*70)