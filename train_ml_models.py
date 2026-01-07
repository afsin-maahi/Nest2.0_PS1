import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("="*80)
print("🤖 TRAINING ML MODELS FOR RISK PREDICTION")
print("="*80)

# Load data
df = pd.read_csv('outputs/all_studies_smart_dqi.csv')

print(f"\n📊 Dataset: {len(df)} patients")
print(f"   Risk distribution:")
print(df['Smart_Risk'].value_counts())

# Prepare features
X = df[['Overdue_Visits_Count', 'Missing_Pages', 'SAE_Pending_Count']].values
y = df['Smart_Risk'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📋 Train set: {len(X_train)} | Test set: {len(X_test)}")

# Model 1: Random Forest
print("\n" + "="*80)
print("🌲 TRAINING RANDOM FOREST")
print("="*80)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"✅ Random Forest Accuracy: {rf_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("\nFeature Importance:")
features = ['Overdue_Visits', 'Missing_Pages', 'SAE_Pending']
for feat, imp in zip(features, rf_model.feature_importances_):
    print(f"  {feat:20s}: {imp:.3f}")

# Model 2: Gradient Boosting
print("\n" + "="*80)
print("🚀 TRAINING GRADIENT BOOSTING")
print("="*80)

gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, y_pred_gb)

print(f"✅ Gradient Boosting Accuracy: {gb_accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_gb))

# Save models
joblib.dump(rf_model, 'outputs/random_forest_model.pkl')
joblib.dump(gb_model, 'outputs/gradient_boosting_model.pkl')

print("\n" + "="*80)
print("📊 MODEL COMPARISON")
print("="*80)
print(f"Random Forest Accuracy:      {rf_accuracy:.3f}")
print(f"Gradient Boosting Accuracy:  {gb_accuracy:.3f}")

# Predict on entire dataset
df['RF_Predicted_Risk'] = rf_model.predict(X)
df['GB_Predicted_Risk'] = gb_model.predict(X)

# Save predictions
df.to_csv('outputs/all_studies_with_ml_predictions.csv', index=False)

print(f"\n💾 Models saved:")
print(f"   → outputs/random_forest_model.pkl")
print(f"   → outputs/gradient_boosting_model.pkl")
print(f"   → outputs/all_studies_with_ml_predictions.csv")

print("\n" + "="*80)
print("✅ ML TRAINING COMPLETE!")
print("="*80)
