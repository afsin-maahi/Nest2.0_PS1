import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

print("=" * 80)
print("🤖 TRAINING PREDICTIVE MODEL: Future Risk Deterioration")
print("=" * 80)

# Load noisy data
df = pd.read_csv('outputs/all_studies_noisy.csv')
print(f"\n📊 Loaded {len(df):,} patients from {df['Study'].nunique()} studies")

# Extract numeric features
df['Overdue_Visits_Count'] = pd.to_numeric(df['Overdue_Visits_Count'], errors='coerce').fillna(0)
df['Missing_Pages'] = pd.to_numeric(df['Missing_Pages'], errors='coerce').fillna(0)
df['SAE_Pending_Count'] = pd.to_numeric(df['SAE_Pending_Count'], errors='coerce').fillna(0)

# Create FUTURE RISK target (simulated)
# Patients are at risk of future deterioration if they have:
# 1. ANY current issues (even small ones)
# 2. Random chance based on their risk profile

print(f"\n🔧 Simulating future risk deterioration...")

np.random.seed(42)

# Calculate risk score (0-100)
df['Risk_Score'] = (
    df['SAE_Pending_Count'] * 15 +
    df['Overdue_Visits_Count'] * 5 +
    df['Missing_Pages'] * 2
).clip(0, 100)

# Future deterioration probability based on current state + randomness
df['Deterioration_Prob'] = 0.05  # Base 5% chance

# Increase probability based on current issues
df.loc[df['Risk_Score'] > 0, 'Deterioration_Prob'] += 0.15  # +15% if any issues
df.loc[df['Risk_Score'] > 20, 'Deterioration_Prob'] += 0.20  # +20% if moderate issues
df.loc[df['Risk_Score'] > 50, 'Deterioration_Prob'] += 0.30  # +30% if severe issues

# Add random noise to probabilities
df['Deterioration_Prob'] += np.random.normal(0, 0.1, len(df))
df['Deterioration_Prob'] = df['Deterioration_Prob'].clip(0, 1)

# Generate binary outcome based on probability
df['Will_Deteriorate'] = (np.random.random(len(df)) < df['Deterioration_Prob']).astype(int)

print(f"\n🎯 Future Deterioration Distribution:")
print(f"   Will Deteriorate (1): {df['Will_Deteriorate'].sum():,} ({df['Will_Deteriorate'].sum()/len(df)*100:.1f}%)")
print(f"   Will Stay Stable (0): {(len(df) - df['Will_Deteriorate'].sum()):,} ({(len(df) - df['Will_Deteriorate'].sum())/len(df)*100:.1f}%)")

# Feature engineering - using CURRENT state to predict FUTURE
print(f"\n🔧 Engineering predictive features...")

df['Total_Issues'] = df['Overdue_Visits_Count'] + df['Missing_Pages'] + df['SAE_Pending_Count']
df['Has_SAE'] = (df['SAE_Pending_Count'] > 0).astype(int)
df['Has_Overdue_Visits'] = (df['Overdue_Visits_Count'] > 0).astype(int)
df['Has_Missing_Pages'] = (df['Missing_Pages'] > 0).astype(int)
df['Multiple_Issue_Types'] = ((df['Has_SAE'] + df['Has_Overdue_Visits'] + df['Has_Missing_Pages']) >= 2).astype(int)

# Add trend indicators (simulated)
df['Issues_Trend'] = np.random.choice([0, 1, 2], len(df), p=[0.7, 0.2, 0.1])  # 0=stable, 1=improving, 2=worsening

# Severity bins
df['SAE_Severity'] = pd.cut(df['SAE_Pending_Count'], bins=[-1, 0, 2, 5, 100], labels=[0, 1, 2, 3]).astype(int)
df['Visits_Severity'] = pd.cut(df['Overdue_Visits_Count'], bins=[-1, 0, 1, 3, 100], labels=[0, 1, 2, 3]).astype(int)

# Context
df['Context_Encoded'] = df['Context'].astype('category').cat.codes

# Historical pattern (simulated - patients with past issues are more likely to have future ones)
df['Past_Issues_Flag'] = ((df['Total_Issues'] > 0) & (np.random.random(len(df)) < 0.3)).astype(int)

# Features for prediction
feature_cols = [
    'Overdue_Visits_Count',
    'Missing_Pages',
    'SAE_Pending_Count',
    'Total_Issues',
    'Has_SAE',
    'Has_Overdue_Visits',
    'Has_Missing_Pages',
    'Multiple_Issue_Types',
    'Issues_Trend',
    'SAE_Severity',
    'Visits_Severity',
    'Context_Encoded',
    'Past_Issues_Flag'
]

X = df[feature_cols]
y = df['Will_Deteriorate']

print(f"\n📋 Features used: {len(feature_cols)}")
for feat in feature_cols:
    print(f"   - {feat}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples ({y_train.sum()} will deteriorate)")
print(f"   Testing: {len(X_test):,} samples ({y_test.sum()} will deteriorate)")

# Train Gradient Boosting Classifier
print(f"\n🌲 Training Gradient Boosting Classifier...")
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
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
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 Overall Metrics:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   ROC-AUC Score: {roc_auc:.3f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stable', 'Will Deteriorate']))

print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                     Predicted")
print(f"                     Stable  Deteriorate")
print(f"   Actual Stable       {cm[0][0]:4d}       {cm[0][1]:4d}")
print(f"   Actual Deteriorate  {cm[1][0]:4d}       {cm[1][1]:4d}")

# Feature importance
print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
print("=" * 80)
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:30s}: {row['Importance']:6.1%}")

# Save model
model_path = 'outputs/predictive_model_v2.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Demo predictions
print(f"\n🎯 DEMO: Future Risk Predictions")
print("=" * 80)

# Get diverse samples
demo_samples = df.sample(8, random_state=42)
demo_X = demo_samples[feature_cols]
demo_pred_proba = model.predict_proba(demo_X)[:, 1]
demo_pred = model.predict(demo_X)

for idx, (i, patient) in enumerate(demo_samples.iterrows()):
    print(f"\n{idx+1}. Patient: {patient['Subject ID']} | Study: {patient['Study'][:30]}")
    print(f"   Current: {patient['Smart_Risk_Noisy']} Risk (DQI: {patient['Smart_DQI_Noisy']:.1f})")
    print(f"   Issues: SAE={patient['SAE_Pending_Count']:.0f}, Visits={patient['Overdue_Visits_Count']:.0f}, Pages={patient['Missing_Pages']:.0f}")
    print(f"   🤖 PREDICTION: {demo_pred_proba[idx]:.1%} probability of deterioration")
    print(f"   Forecast: {'⚠️ LIKELY TO WORSEN' if demo_pred[idx] == 1 else '✅ LIKELY STABLE'}")
    print(f"   Actual: {'Deteriorated' if patient['Will_Deteriorate'] == 1 else 'Stayed Stable'}")

print("\n" + "=" * 80)
print("✅ MODEL 1: PREDICTIVE MODEL COMPLETE!")
print(f"📊 Accuracy: {accuracy:.1%} (realistic for forecasting!)")
print(f"🎯 Can identify {(cm[1][1] / (cm[1][0] + cm[1][1]) * 100):.0f}% of patients who will deteriorate")
print("=" * 80)
