import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib

print("=" * 80)
print("🤖 TRAINING PREDICTIVE MODEL: Future Data Quality Issues")
print("=" * 80)

# Load data
df = pd.read_csv('outputs/all_studies_noisy.csv')

print(f"\n📊 Loaded {len(df):,} patients from {df['Study'].nunique()} studies")

# Create target variable: Will patient have SEVERE issues?
# Define "severe issues" as High risk OR multiple problems
df['Overdue_Visits_Count'] = pd.to_numeric(df['Overdue_Visits_Count'], errors='coerce').fillna(0)
df['Missing_Pages'] = pd.to_numeric(df['Missing_Pages'], errors='coerce').fillna(0)
df['SAE_Pending_Count'] = pd.to_numeric(df['SAE_Pending_Count'], errors='coerce').fillna(0)

# Target: Severe issues = High risk patients
df['Severe_Issues'] = (df['Smart_Risk_Noisy'] == 'High').astype(int)

print(f"\n🎯 Target Distribution:")
print(f"   Severe Issues (1): {df['Severe_Issues'].sum():,} ({df['Severe_Issues'].sum()/len(df)*100:.1f}%)")
print(f"   Normal (0): {(len(df) - df['Severe_Issues'].sum()):,} ({(len(df) - df['Severe_Issues'].sum())/len(df)*100:.1f}%)")

# Feature engineering - ONLY use raw metrics (no DQI scores!)
print(f"\n🔧 Engineering features (NO DQI LEAKAGE)...")

# Create derived features from raw data only
df['Total_Issues'] = df['Overdue_Visits_Count'] + df['Missing_Pages'] + df['SAE_Pending_Count']
df['Has_SAE'] = (df['SAE_Pending_Count'] > 0).astype(int)
df['Has_Overdue_Visits'] = (df['Overdue_Visits_Count'] > 0).astype(int)
df['Has_Missing_Pages'] = (df['Missing_Pages'] > 0).astype(int)
df['Multiple_Issue_Types'] = (df['Has_SAE'] + df['Has_Overdue_Visits'] + df['Has_Missing_Pages']).clip(0, 1)

# SAE severity indicators
df['SAE_Critical'] = (df['SAE_Pending_Count'] >= 3).astype(int)
df['SAE_Very_Critical'] = (df['SAE_Pending_Count'] >= 5).astype(int)

# Visit delay severity
df['Visits_Critical'] = (df['Overdue_Visits_Count'] >= 2).astype(int)

# Interaction features
df['SAE_x_Visits'] = df['SAE_Pending_Count'] * df['Overdue_Visits_Count']
df['SAE_x_Pages'] = df['SAE_Pending_Count'] * df['Missing_Pages']

# Encode context
df['Context_Encoded'] = df['Context'].astype('category').cat.codes

# Select features - NO DQI SCORES ALLOWED!
feature_cols = [
    'Overdue_Visits_Count',
    'Missing_Pages',
    'SAE_Pending_Count',
    'Total_Issues',
    'Has_SAE',
    'Has_Overdue_Visits',
    'Has_Missing_Pages',
    'Multiple_Issue_Types',
    'SAE_Critical',
    'SAE_Very_Critical',
    'Visits_Critical',
    'SAE_x_Visits',
    'SAE_x_Pages',
    'Context_Encoded'
]

X = df[feature_cols]
y = df['Severe_Issues']

print(f"\n📋 Features used: {len(feature_cols)}")
for feat in feature_cols:
    print(f"   - {feat}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples ({y_train.sum()} severe, {len(y_train)-y_train.sum()} normal)")
print(f"   Testing: {len(X_test):,} samples ({y_test.sum()} severe, {len(y_test)-y_test.sum()} normal)")

# Train Gradient Boosting model (better for imbalanced data)
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

# Evaluate on test set
print(f"\n📊 MODEL PERFORMANCE:")
print("=" * 80)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n🎯 Overall Metrics:")
print(f"   Accuracy: {accuracy:.1%}")
print(f"   ROC-AUC Score: {roc_auc:.3f}")

print(f"\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Severe Issues']))

print(f"\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"                     Predicted")
print(f"                     Normal  Severe")
print(f"   Actual Normal       {cm[0][0]:4d}    {cm[0][1]:4d}")
print(f"   Actual Severe       {cm[1][0]:4d}    {cm[1][1]:4d}")

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
model_path = 'outputs/predictive_model.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save feature list for later use
feature_info = {
    'features': feature_cols,
    'target': 'Severe_Issues'
}
joblib.dump(feature_info, 'outputs/model_features.pkl')
print(f"💾 Feature info saved to: outputs/model_features.pkl")

# Demo predictions on diverse patients
print(f"\n🎯 DEMO: Predictions on Sample Patients")
print("=" * 80)

# Get diverse samples
high_risk = df[df['Severe_Issues'] == 1].sample(min(3, df['Severe_Issues'].sum()), random_state=42)
low_risk = df[df['Severe_Issues'] == 0].sample(3, random_state=42)
demo_patients = pd.concat([high_risk, low_risk])

demo_X = demo_patients[feature_cols]
demo_pred_proba = model.predict_proba(demo_X)[:, 1]
demo_pred = model.predict(demo_X)

for idx, (i, patient) in enumerate(demo_patients.iterrows()):
    print(f"\n{idx+1}. Patient: {patient['Subject ID']} | Study: {patient['Study'][:30]}")
    print(f"   Current Status: {patient['Smart_Risk']} Risk (DQI: {patient['Smart_DQI']:.1f})")
    print(f"   SAE: {patient['SAE_Pending_Count']:.0f} | Visits: {patient['Overdue_Visits_Count']:.0f} | Pages: {patient['Missing_Pages']:.0f}")
    print(f"   🤖 PREDICTION: {demo_pred_proba[idx]:.1%} probability of severe issues")
    print(f"   Risk Level: {'🚨 SEVERE ISSUES LIKELY' if demo_pred[idx] == 1 else '✅ NORMAL'}")

print("\n" + "=" * 80)
print("✅ MODEL 1 TRAINING COMPLETE!")
print(f"📊 Realistic Accuracy: {accuracy:.1%} (should be 85-95%)")
print("=" * 80)
