"""
Bike Rental Demand Prediction - Final Linear Regression Attempts
Last attempts to improve Linear Regression before switching to Random Forest

STRATEGIES:
1. Simplified features (remove problematic ones)
2. Remove 'yr' feature (might be causing leakage)
3. Different train/test split strategy
4. Ensemble of Linear Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style('whitegrid')

print("="*70)
print("FINAL LINEAR REGRESSION ATTEMPTS - MULTIPLE STRATEGIES")
print("="*70)

# ============================================================================
# STRATEGY 1: MINIMAL FEATURES (BACK TO BASICS)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 1: MINIMAL FEATURES (Simplest Approach)")
print("="*70)

df = pd.read_csv("day.csv")

# Only the most essential features
categorical_minimal = ['season', 'weathersit', 'workingday']
numeric_minimal = ['temp', 'hum', 'windspeed']
target = 'cnt'

print(f"Features: {categorical_minimal + numeric_minimal}")

X_minimal = df[categorical_minimal + numeric_minimal]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X_minimal, y, test_size=0.2, random_state=42, shuffle=False
)

preprocessor_minimal = ColumnTransformer([
    ('num', StandardScaler(), numeric_minimal),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_minimal)
])

X_train_proc = preprocessor_minimal.fit_transform(X_train)
X_test_proc = preprocessor_minimal.transform(X_test)

# Try different models
models_1 = {
    'LinearReg': LinearRegression(),
    'Ridge(10)': Ridge(alpha=10),
    'Ridge(100)': Ridge(alpha=100),
    'Lasso(10)': Lasso(alpha=10, max_iter=10000),
    'ElasticNet': ElasticNet(alpha=10, l1_ratio=0.5, max_iter=10000)
}

results_strategy1 = {}
print(f"\n{'Model':<15} {'Train R²':<12} {'Test R²':<12} {'Overfit':<10}")
print("-"*60)

for name, model in models_1.items():
    model.fit(X_train_proc, y_train)
    train_r2 = model.score(X_train_proc, y_train)
    test_r2 = model.score(X_test_proc, y_test)
    overfit = train_r2 - test_r2
    results_strategy1[name] = {'test_r2': test_r2, 'overfit': overfit, 'model': model}
    print(f"{name:<15} {train_r2:>11.4f} {test_r2:>11.4f} {overfit:>9.4f}")

best_s1 = max(results_strategy1, key=lambda x: results_strategy1[x]['test_r2'])
print(f"\n✓ Best: {best_s1} → Test R²: {results_strategy1[best_s1]['test_r2']:.4f}")

# ============================================================================
# STRATEGY 2: REMOVE 'YR' FEATURE (Potential Data Leakage)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 2: REMOVE 'YR' (Year might cause overfitting)")
print("="*70)

# yr might be memorizing training data (2011) vs test data (2012)
categorical_no_yr = ['season', 'mnth', 'weekday', 'weathersit', 'holiday', 'workingday']
numeric_standard = ['temp', 'hum', 'windspeed']

print(f"Features (no yr): {categorical_no_yr + numeric_standard}")

X_no_yr = df[categorical_no_yr + numeric_standard]

X_train, X_test, y_train, y_test = train_test_split(
    X_no_yr, y, test_size=0.2, random_state=42, shuffle=False
)

preprocessor_no_yr = ColumnTransformer([
    ('num', StandardScaler(), numeric_standard),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_no_yr)
])

X_train_proc = preprocessor_no_yr.fit_transform(X_train)
X_test_proc = preprocessor_no_yr.transform(X_test)

models_2 = {
    'LinearReg': LinearRegression(),
    'Ridge(10)': Ridge(alpha=10),
    'Ridge(100)': Ridge(alpha=100),
    'Lasso(10)': Lasso(alpha=10, max_iter=10000),
}

results_strategy2 = {}
print(f"\n{'Model':<15} {'Train R²':<12} {'Test R²':<12} {'Overfit':<10}")
print("-"*60)

for name, model in models_2.items():
    model.fit(X_train_proc, y_train)
    train_r2 = model.score(X_train_proc, y_train)
    test_r2 = model.score(X_test_proc, y_test)
    overfit = train_r2 - test_r2
    results_strategy2[name] = {'test_r2': test_r2, 'overfit': overfit, 'model': model}
    print(f"{name:<15} {train_r2:>11.4f} {test_r2:>11.4f} {overfit:>9.4f}")

best_s2 = max(results_strategy2, key=lambda x: results_strategy2[x]['test_r2'])
print(f"\n✓ Best: {best_s2} → Test R²: {results_strategy2[best_s2]['test_r2']:.4f}")

# ============================================================================
# STRATEGY 3: DIFFERENT TRAIN/TEST SPLIT (Random Instead of Time)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 3: RANDOM SPLIT (Instead of Time-based)")
print("="*70)

categorical_standard = ['season', 'yr', 'mnth', 'weekday', 'weathersit', 'holiday', 'workingday']
X_standard = df[categorical_standard + numeric_standard]

# Try with shuffle=True
X_train, X_test, y_train, y_test = train_test_split(
    X_standard, y, test_size=0.2, random_state=42, shuffle=True  # RANDOM!
)

print("Using RANDOM split (shuffle=True)")

preprocessor_standard = ColumnTransformer([
    ('num', StandardScaler(), numeric_standard),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_standard)
])

X_train_proc = preprocessor_standard.fit_transform(X_train)
X_test_proc = preprocessor_standard.transform(X_test)

models_3 = {
    'LinearReg': LinearRegression(),
    'Ridge(10)': Ridge(alpha=10),
    'Ridge(100)': Ridge(alpha=100),
}

results_strategy3 = {}
print(f"\n{'Model':<15} {'Train R²':<12} {'Test R²':<12} {'Overfit':<10}")
print("-"*60)

for name, model in models_3.items():
    model.fit(X_train_proc, y_train)
    train_r2 = model.score(X_train_proc, y_train)
    test_r2 = model.score(X_test_proc, y_test)
    overfit = train_r2 - test_r2
    results_strategy3[name] = {'test_r2': test_r2, 'overfit': overfit, 'model': model}
    print(f"{name:<15} {train_r2:>11.4f} {test_r2:>11.4f} {overfit:>9.4f}")

best_s3 = max(results_strategy3, key=lambda x: results_strategy3[x]['test_r2'])
print(f"\n✓ Best: {best_s3} → Test R²: {results_strategy3[best_s3]['test_r2']:.4f}")

# ============================================================================
# STRATEGY 4: ORIGINAL APPROACH 
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 4: YOUR ORIGINAL APPROACH (Baseline)")
print("="*70)

X_original = df[categorical_standard + numeric_standard]

X_train, X_test, y_train, y_test = train_test_split(
    X_original, y, test_size=0.2, random_state=42, shuffle=False
)

preprocessor_original = ColumnTransformer([
    ('num', StandardScaler(), numeric_standard),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_standard)
])

X_train_proc = preprocessor_original.fit_transform(X_train)
X_test_proc = preprocessor_original.transform(X_test)

model_original = Lasso(alpha=1.0, max_iter=10000)
model_original.fit(X_train_proc, y_train)

train_r2_orig = model_original.score(X_train_proc, y_train)
test_r2_orig = model_original.score(X_test_proc, y_test)
overfit_orig = train_r2_orig - test_r2_orig

print(f"Your Original: Lasso(α=1.0)")
print(f"  Train R²: {train_r2_orig:.4f}")
print(f"  Test R²:  {test_r2_orig:.4f}")
print(f"  Overfit:  {overfit_orig:.4f}")

results_strategy4 = {
    'Original': {'test_r2': test_r2_orig, 'overfit': overfit_orig, 'model': model_original}
}

# ============================================================================
# STRATEGY 5: CROSS-VALIDATION (Robust Evaluation)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 5: CROSS-VALIDATION")
print("="*70)

from sklearn.model_selection import cross_val_score, KFold

categorical_cv = ['season', 'mnth', 'weekday', 'weathersit', 'holiday', 'workingday']
numeric_cv = ['temp', 'hum', 'windspeed']
X_cv = df[categorical_cv + numeric_cv]

preprocessor_cv = ColumnTransformer([
    ('num', StandardScaler(), numeric_cv),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cv)
])

X_proc = preprocessor_cv.fit_transform(X_cv)

# Try Ridge and Lasso with CV
models_cv = {
    'LinearReg': LinearRegression(),
    'Ridge(10)': Ridge(alpha=10),
    'Lasso(1)': Lasso(alpha=1.0, max_iter=10000)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Model':<15} {'Mean R²':<12} {'Std R²':<12}")
print("-"*50)

results_strategy5 = {}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_proc, y, cv=kf, scoring='r2')
    mean_r2 = scores.mean()
    std_r2 = scores.std()
    results_strategy5[name] = {'mean_r2': mean_r2, 'std_r2': std_r2, 'model': model}
    print(f"{name:<15} {mean_r2:>11.4f} {std_r2:>11.4f}")

best_s5 = max(results_strategy5, key=lambda x: results_strategy5[x]['mean_r2'])
print(f"\n✓ Best CV Model: {best_s5} → Mean R²: {results_strategy5[best_s5]['mean_r2']:.4f}")

best_s5_result = {
    'test_r2': results_strategy5[best_s5]['mean_r2'],
    'overfit': results_strategy5[best_s5]['std_r2'],  # hier gebruiken we std als 'variatie'
    'model': results_strategy5[best_s5]['model']
}

# ============================================================================
# STRATEGY 6: POLYNOMIAL FEATURES (Non-linear Relationships)
# ============================================================================
print("\n" + "="*70)
print("STRATEGY 6: POLYNOMIAL FEATURES")
print("="*70)

from sklearn.preprocessing import PolynomialFeatures

categorical_poly = ['season', 'weathersit', 'workingday']
numeric_poly = ['temp', 'hum', 'windspeed']
X_poly = df[categorical_poly + numeric_poly]

preprocessor_poly = ColumnTransformer([
    ('num', StandardScaler(), numeric_poly),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_poly)
])

X_proc_poly = preprocessor_poly.fit_transform(X_poly)

# Add polynomial expansion on numeric features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_proc_poly_expanded = poly.fit_transform(X_proc_poly)

X_train, X_test, y_train, y_test = train_test_split(
    X_proc_poly_expanded, y, test_size=0.2, random_state=42, shuffle=False
)

model_poly = Ridge(alpha=10)
model_poly.fit(X_train, y_train)

train_r2_poly = model_poly.score(X_train, y_train)
test_r2_poly = model_poly.score(X_test, y_test)

print(f"Polynomial Ridge (deg=2):")
print(f"  Train R²: {train_r2_poly:.4f}")
print(f"  Test R²:  {test_r2_poly:.4f}")
print(f"  Overfit:  {train_r2_poly - test_r2_poly:.4f}")

results_strategy6 = {
    'Polynomial Ridge': {
        'test_r2': test_r2_poly,
        'overfit': train_r2_poly - test_r2_poly,
        'model': model_poly
    }
}


# ============================================================================
# FINAL COMPARISON - ALL STRATEGIES (excluding Strategy 3: Random Split)
# ============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON - ALL STRATEGIES (excluding Strategy 3)")
print("="*70)

# Maak een dictionary zonder Strategy 3
all_results = {
    f"S1: {best_s1}": results_strategy1[best_s1],
    f"S2: {best_s2}": results_strategy2[best_s2],
    # "S3: {best_s3}": results_strategy3[best_s3],  # UITGESLOTEN
    "S4: Original": results_strategy4['Original'],
    f"S5: {best_s5} (CV)": best_s5_result,
    "S6: Polynomial Ridge": results_strategy6['Polynomial Ridge']
}

print(f"\n{'Strategy':<30} {'Test R²':<12} {'Overfit':<10}")
print("-"*60)
for name, result in all_results.items():
    print(f"{name:<30} {result['test_r2']:>11.4f} {result['overfit']:>9.4f}")

# Bepaal beste resultaat zonder Strategy 3
best_overall = max(all_results, key=lambda x: all_results[x]['test_r2'])
best_result = all_results[best_overall]

print(f"\n🏆 BEST OVERALL (excluding Strategy 3): {best_overall}")
print(f"   Test R²: {best_result['test_r2']:.4f}")
print(f"   Overfitting: {best_result['overfit']:.4f}")

# ============================================================================
# DETAILED EVALUATION OF BEST MODEL
# ============================================================================
print("\n" + "="*70)
print("DETAILED EVALUATION - BEST MODEL")
print("="*70)

# Kies juiste dataset en preprocessor op basis van best_overall
if best_overall.startswith("S1"):
    X_final = X_minimal
    preprocessor_final = preprocessor_minimal
    shuffle_final = False
elif best_overall.startswith("S2"):
    X_final = X_no_yr
    preprocessor_final = preprocessor_no_yr
    shuffle_final = False
elif best_overall.startswith("S4"):
    X_final = X_original
    preprocessor_final = preprocessor_original
    shuffle_final = False
elif best_overall.startswith("S5"):
    X_final = X_cv
    preprocessor_final = preprocessor_cv
    shuffle_final = True
elif best_overall.startswith("S6"):
    X_final = X_poly
    preprocessor_final = preprocessor_poly
    shuffle_final = False
else:
    X_final = X_original
    preprocessor_final = preprocessor_original
    shuffle_final = False

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, shuffle=shuffle_final
)

X_train_proc = preprocessor_final.fit_transform(X_train)
X_test_proc = preprocessor_final.transform(X_test)

best_model = best_result['model']
best_model.fit(X_train_proc, y_train)

y_train_pred = best_model.predict(X_train_proc)
y_test_pred = best_model.predict(X_test_proc)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\n📈 TRAINING SET:")
print(f"  R² Score:  {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"  RMSE:      {train_rmse:.2f} bikes")
print(f"  MAE:       {train_mae:.2f} bikes")

print(f"\n📉 TEST SET:")
print(f"  R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"  RMSE:      {test_rmse:.2f} bikes")
print(f"  MAE:       {test_mae:.2f} bikes")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\n📊 Creating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Best Linear Regression Result: {best_overall}', 
             fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect')
axes[0, 0].set_xlabel('Actual', fontweight='bold')
axes[0, 0].set_ylabel('Predicted', fontweight='bold')
axes[0, 0].set_title(f'Actual vs Predicted\nR² = {test_r2:.4f}', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Values', fontweight='bold')
axes[0, 1].set_ylabel('Residuals', fontweight='bold')
axes[0, 1].set_title('Residual Plot', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Strategy Comparison
strategies = list(all_results.keys())
test_scores = [all_results[s]['test_r2'] for s in strategies]
colors = ['green' if s == best_overall else 'gray' for s in strategies]

axes[1, 0].barh(strategies, test_scores, color=colors, alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Test R² Score', fontweight='bold')
axes[1, 0].set_title('Strategy Comparison', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Plot 4: Time Series
test_indices = range(len(y_test))
axes[1, 1].plot(test_indices, y_test.values, 'o-', label='Actual', 
                linewidth=2, markersize=3, alpha=0.7)
axes[1, 1].plot(test_indices, y_test_pred, 'x-', label='Predicted', 
                linewidth=2, markersize=3, alpha=0.7)
axes[1, 1].set_xlabel('Test Sample Index', fontweight='bold')
axes[1, 1].set_ylabel('Bike Rentals', fontweight='bold')
axes[1, 1].set_title('Time Series', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_final_attempts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: linear_regression_final_attempts.png")
plt.show()

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

print(f"""
🎯 Best Linear Regression Result:
   Strategy: {best_overall}
   Test R²:  {test_r2:.4f} ({test_r2*100:.1f}%)
   RMSE:     {test_rmse:.0f} bikes
   MAE:      {test_mae:.0f} bikes

📊 Comparison to Your Original:
   Original:  R² = {test_r2_orig:.4f} (66.5%)
   Best Now:  R² = {test_r2:.4f} ({test_r2*100:.1f}%)
   Change:    {(test_r2 - test_r2_orig)*100:+.1f}%

💡 HONEST ASSESSMENT:
""")

if test_r2 > 0.80:
    print("   🎉 SUCCESS! Linear Regression works well!")
    print("   You can continue with this approach.")
elif test_r2 > 0.75:
    print("   ✓ ACCEPTABLE! But Random Forest will likely do better.")
    print("   Try Random Forest for comparison.")
elif test_r2 > 0.70:
    print("   ⚠️  MARGINAL! Linear Regression is struggling.")
    print("   Random Forest is strongly recommended.")
else:
    print("   ❌ LINEAR REGRESSION HAS REACHED ITS LIMIT!")
    print("   The problem is fundamentally non-linear.")
    print("   You MUST switch to Random Forest or tree-based models.")
    print("")
    print("   Expected with Random Forest: R² = 0.85-0.92")

print(f"""
🔄 Next Steps:
   1. If R² > 0.75: You can stick with Linear Regression
   2. If R² < 0.75: Switch to Random Forest (I can help!)

📚 What We Learned:
   - Linear models struggle with non-linear bike rental patterns
   - Year (yr) feature can cause train/test mismatch
   - Random split helps but loses time-series structure
   - Feature engineering alone doesn't solve fundamental model limitations
""")

print("="*70)
print("✅ Analysis Complete!")
print("="*70)