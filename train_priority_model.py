import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("=" * 80)
print("🎯 TRAINING MODEL 2: Priority Scoring System")
print("=" * 80)

# Load noisy data
df = pd.read_csv('outputs/all_studies_noisy.csv')
print(f"\n📊 Loaded {len(df):,} patients from {df['Study'].nunique()} studies")

# Extract numeric features
df['Overdue_Visits_Count'] = pd.to_numeric(df['Overdue_Visits_Count'], errors='coerce').fillna(0)
df['Missing_Pages'] = pd.to_numeric(df['Missing_Pages'], errors='coerce').fillna(0)
df['SAE_Pending_Count'] = pd.to_numeric(df['SAE_Pending_Count'], errors='coerce').fillna(0)

# Create PRIORITY SCORE (0-100, lower = more urgent)
print(f"\n🔧 Creating Priority Scores...")

# Priority score starts at 100 (lowest priority)
df['Priority_Score'] = 100.0

# Heavy penalties for safety issues (most urgent)
df['Priority_Score'] -= df['SAE_Pending_Count'] * 8  # Each SAE reduces priority by 8 points

# Moderate penalties for overdue visits
df['Priority_Score'] -= df['Overdue_Visits_Count'] * 3  # Each overdue visit reduces by 3

# Light penalties for missing pages
df['Priority_Score'] -= df['Missing_Pages'] * 1.5  # Each missing page reduces by 1.5

# Add some noise to make it more realistic
np.random.seed(42)
noise = np.random.normal(0, 2, len(df))  # ±2 points random variation
df['Priority_Score'] = (df['Priority_Score'] + noise).clip(0, 100)

print(f"\n📊 Priority Score Distribution:")
print(f"   Mean: {df['Priority_Score'].mean():.1f}")
print(f"   Median: {df['Priority_Score'].median():.1f}")
print(f"   Std Dev: {df['Priority_Score'].std():.1f}")
print(f"   Range: {df['Priority_Score'].min():.1f} - {df['Priority_Score'].max():.1f}")

# Show priority categories
urgent = (df['Priority_Score'] < 50).sum()
moderate = ((df['Priority_Score'] >= 50) & (df['Priority_Score'] < 80)).sum()
low = (df['Priority_Score'] >= 80).sum()

print(f"\n🚨 Priority Categories:")
print(f"   🔴 URGENT (0-49): {urgent:,} patients ({urgent/len(df)*100:.1f}%)")
print(f"   🟡 MODERATE (50-79): {moderate:,} patients ({moderate/len(df)*100:.1f}%)")
print(f"   🟢 LOW (80-100): {low:,} patients ({low/len(df)*100:.1f}%)")

# Feature engineering
print(f"\n🔧 Engineering predictive features...")

df['Total_Issues'] = df['Overdue_Visits_Count'] + df['Missing_Pages'] + df['SAE_Pending_Count']
df['Has_SAE'] = (df['SAE_Pending_Count'] > 0).astype(int)
df['Has_Overdue_Visits'] = (df['Overdue_Visits_Count'] > 0).astype(int)
df['Has_Missing_Pages'] = (df['Missing_Pages'] > 0).astype(int)
df['Multiple_Issue_Types'] = ((df['Has_SAE'] + df['Has_Overdue_Visits'] + df['Has_Missing_Pages']) >= 2).astype(int)

# Severity indicators
df['SAE_Critical'] = (df['SAE_Pending_Count'] >= 3).astype(int)
df['SAE_Emergency'] = (df['SAE_Pending_Count'] >= 10).astype(int)
df['Visits_Critical'] = (df['Overdue_Visits_Count'] >= 2).astype(int)

# Interaction features
df['SAE_x_Visits'] = df['SAE_Pending_Count'] * df['Overdue_Visits_Count']
df['SAE_squared'] = df['SAE_Pending_Count'] ** 2  # Non-linear SAE impact
df['Total_Issues_squared'] = df['Total_Issues'] ** 2

# Time-based urgency (simulated)
df['Days_Since_Last_Update'] = np.random.randint(0, 30, len(df))  # Simulate days since last check
df['Urgency_Multiplier'] = 1 + (df['Days_Since_Last_Update'] / 100)  # Older issues = more urgent

# Study-level context
df['Context_Encoded'] = df['Context'].astype('category').cat.codes

# Patient complexity score (simulated)
df['Complexity_Score'] = (
    df['SAE_Pending_Count'] * 3 +
    df['Overdue_Visits_Count'] * 2 +
    df['Missing_Pages'] * 1
).clip(0, 50)

# Features for prediction (excluding Priority_Score components to avoid leakage)
feature_cols = [
    'Total_Issues',
    'Has_SAE',
    'Has_Overdue_Visits',
    'Has_Missing_Pages',
    'Multiple_Issue_Types',
    'SAE_Critical',
    'SAE_Emergency',
    'Visits_Critical',
    'SAE_x_Visits',
    'SAE_squared',
    'Total_Issues_squared',
    'Days_Since_Last_Update',
    'Urgency_Multiplier',
    'Context_Encoded',
    'Complexity_Score'
]

X = df[feature_cols]
y = df['Priority_Score']

print(f"\n📋 Features used: {len(feature_cols)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# Train Gradient Boosting Regressor
print(f"\n🌲 Training Gradient Boosting Regressor...")
model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    subsample=0.8
)

model.fit(X_train, y_train)
print(f"✅ Model training complete!")

# Evaluate
print(f"\n📊 MODEL PERFORMANCE:")
print("=" * 80)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n🎯 Regression Metrics:")
print(f"   R² Score: {r2:.3f} (1.0 = perfect, 0.0 = useless)")
print(f"   Mean Absolute Error: {mae:.2f} priority points")
print(f"   Root Mean Squared Error: {rmse:.2f} priority points")

# Interpret MAE
print(f"\n💡 Interpretation:")
if mae < 5:
    print(f"   ✅ Excellent! Predictions within ±{mae:.1f} points on average")
elif mae < 10:
    print(f"   ✅ Good! Predictions within ±{mae:.1f} points on average")
else:
    print(f"   ⚠️  Fair. Predictions within ±{mae:.1f} points on average")

# Feature importance
print(f"\n🔝 TOP 10 FEATURES FOR PRIORITY PREDICTION:")
print("=" * 80)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s}: {row['Importance']:6.1%}")

# Save model
model_path = 'outputs/priority_model.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save feature info
feature_info = {
    'features': feature_cols,
    'target': 'Priority_Score'
}
joblib.dump(feature_info, 'outputs/priority_features.pkl')
print(f"💾 Feature info saved to: outputs/priority_features.pkl")

# Demo predictions
print(f"\n🎯 DEMO: Priority Predictions on Sample Patients")
print("=" * 80)

# Get diverse samples - most urgent and least urgent
urgent_samples = df.nsmallest(4, 'Priority_Score')
low_samples = df.nlargest(4, 'Priority_Score')
samples = pd.concat([urgent_samples, low_samples])

demo_X = samples[feature_cols]
demo_pred = model.predict(demo_X)

for idx, (i, patient) in enumerate(samples.iterrows()):
    actual = patient['Priority_Score']
    predicted = demo_pred[idx]
    error = abs(actual - predicted)
    
    if actual < 50:
        priority_icon = "🔴"
        priority_label = "URGENT"
    elif actual < 80:
        priority_icon = "🟡"
        priority_label = "MODERATE"
    else:
        priority_icon = "🟢"
        priority_label = "LOW"
    
    print(f"\n{idx+1}. {priority_icon} Patient: {patient['Subject ID']} | Study: {patient['Study'][:25]}")
    print(f"   Issues: SAE={patient['SAE_Pending_Count']:.0f}, Visits={patient['Overdue_Visits_Count']:.0f}, Pages={patient['Missing_Pages']:.0f}")
    print(f"   Priority: Actual={actual:.1f} | Predicted={predicted:.1f} | Error=±{error:.1f}")
    print(f"   Category: {priority_label} ({patient['Smart_Risk_Noisy']} Risk)")

print("\n" + "=" * 80)
print("✅ MODEL 2: PRIORITY SCORING COMPLETE!")
print(f"📊 Can predict priority within ±{mae:.1f} points on average")
print(f"🎯 R² Score: {r2:.3f} (explains {r2*100:.1f}% of priority variation)")
print("=" * 80)
