import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

print("=" * 80)
print("🔍 TRAINING MODEL 3: Anomaly Detection System")
print("=" * 80)

# Load noisy data
df = pd.read_csv('outputs/all_studies_noisy.csv')
print(f"\n📊 Loaded {len(df):,} patients from {df['Study'].nunique()} studies")

# Extract numeric features
df['Overdue_Visits_Count'] = pd.to_numeric(df['Overdue_Visits_Count'], errors='coerce').fillna(0)
df['Missing_Pages'] = pd.to_numeric(df['Missing_Pages'], errors='coerce').fillna(0)
df['SAE_Pending_Count'] = pd.to_numeric(df['SAE_Pending_Count'], errors='coerce').fillna(0)

# Feature engineering for anomaly detection
print(f"\n🔧 Engineering anomaly detection features...")

# Basic metrics
df['Total_Issues'] = df['Overdue_Visits_Count'] + df['Missing_Pages'] + df['SAE_Pending_Count']

# Ratios and patterns
df['SAE_to_Total_Ratio'] = df['SAE_Pending_Count'] / (df['Total_Issues'] + 1)  # +1 to avoid div by 0
df['Visits_to_Total_Ratio'] = df['Overdue_Visits_Count'] / (df['Total_Issues'] + 1)

# Severity indicators
df['SAE_Severity'] = pd.cut(df['SAE_Pending_Count'], bins=[-1, 0, 2, 5, 100], labels=[0, 1, 2, 3]).astype(int)
df['Visits_Severity'] = pd.cut(df['Overdue_Visits_Count'], bins=[-1, 0, 1, 3, 100], labels=[0, 1, 2, 3]).astype(int)

# Unusual combinations
df['High_SAE_No_Visits'] = ((df['SAE_Pending_Count'] > 5) & (df['Overdue_Visits_Count'] == 0)).astype(int)
df['High_Visits_No_SAE'] = ((df['Overdue_Visits_Count'] > 3) & (df['SAE_Pending_Count'] == 0)).astype(int)

# DQI patterns
df['DQI_Gap'] = abs(df['Smart_DQI_Noisy'] - df['Basic_DQI'])
df['Unusually_High_DQI'] = (df['Smart_DQI_Noisy'] > 95).astype(int)
df['Unusually_Low_DQI'] = (df['Smart_DQI_Noisy'] < 70).astype(int)

# Context
df['Context_Encoded'] = df['Context'].astype('category').cat.codes

# Statistical features - how far from mean?
for col in ['SAE_Pending_Count', 'Overdue_Visits_Count', 'Smart_DQI_Noisy']:
    mean = df[col].mean()
    std = df[col].std()
    df[f'{col}_zscore'] = (df[col] - mean) / (std + 0.001)  # Z-score

# Select features for anomaly detection
feature_cols = [
    'Smart_DQI_Noisy',
    'Total_Issues',
    'SAE_Pending_Count',
    'Overdue_Visits_Count',
    'Missing_Pages',
    'SAE_to_Total_Ratio',
    'Visits_to_Total_Ratio',
    'SAE_Severity',
    'Visits_Severity',
    'High_SAE_No_Visits',
    'High_Visits_No_SAE',
    'DQI_Gap',
    'Context_Encoded',
    'SAE_Pending_Count_zscore',
    'Overdue_Visits_Count_zscore',
    'Smart_DQI_Noisy_zscore'
]

X = df[feature_cols].copy()

print(f"\n📋 Features used: {len(feature_cols)}")
for feat in feature_cols:
    print(f"   - {feat}")

# Scale features (important for anomaly detection)
print(f"\n📊 Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print(f"\n🌳 Training Isolation Forest (Anomaly Detector)...")
model = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination=0.05,  # Expect 5% anomalies
    random_state=42,
    n_jobs=-1
)

# Fit model
anomaly_labels = model.fit_predict(X_scaled)
anomaly_scores = model.decision_function(X_scaled)

# Convert labels: -1 (anomaly) to 1, 1 (normal) to 0
df['Is_Anomaly'] = (anomaly_labels == -1).astype(int)
df['Anomaly_Score'] = -anomaly_scores  # Invert so higher = more anomalous

print(f"✅ Model training complete!")

# Results
print(f"\n📊 ANOMALY DETECTION RESULTS:")
print("=" * 80)

n_anomalies = df['Is_Anomaly'].sum()
pct_anomalies = n_anomalies / len(df) * 100

print(f"\n🎯 Detection Summary:")
print(f"   Total Patients: {len(df):,}")
print(f"   Normal: {len(df) - n_anomalies:,} ({100-pct_anomalies:.1f}%)")
print(f"   Anomalies: {n_anomalies:,} ({pct_anomalies:.1f}%)")

# Analyze anomalies by risk category
print(f"\n🔍 Anomalies by Risk Category:")
anomaly_risk = df[df['Is_Anomaly'] == 1]['Smart_Risk_Noisy'].value_counts()
for risk, count in anomaly_risk.items():
    print(f"   {risk} Risk: {count:,} anomalies")

# Anomalies by study
print(f"\n📚 Studies with Most Anomalies:")
anomaly_study = df[df['Is_Anomaly'] == 1]['Study'].value_counts().head(5)
for study, count in anomaly_study.items():
    total_in_study = (df['Study'] == study).sum()
    pct = count / total_in_study * 100
    print(f"   {study[:40]:40s}: {count:3d} ({pct:.1f}%)")

# Feature importance - which features differentiate anomalies?
print(f"\n🔝 ANOMALY CHARACTERISTICS:")
print("=" * 80)

anomaly_df = df[df['Is_Anomaly'] == 1]
normal_df = df[df['Is_Anomaly'] == 0]

print(f"\nAverage values - Anomalies vs Normal:")
key_features = ['SAE_Pending_Count', 'Overdue_Visits_Count', 'Smart_DQI_Noisy', 'Total_Issues']
for feat in key_features:
    anom_mean = anomaly_df[feat].mean()
    norm_mean = normal_df[feat].mean()
    diff = anom_mean - norm_mean
    print(f"   {feat:30s}: Anomaly={anom_mean:6.1f} | Normal={norm_mean:6.1f} | Diff={diff:+6.1f}")

# Save model and scaler
model_path = 'outputs/anomaly_detector.pkl'
scaler_path = 'outputs/anomaly_scaler.pkl'
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"\n💾 Model saved to: {model_path}")
print(f"💾 Scaler saved to: {scaler_path}")

# Save anomaly results
df_with_anomalies = df[['Subject ID', 'Site ID', 'Study', 'Smart_Risk_Noisy', 'Smart_DQI_Noisy', 
                          'SAE_Pending_Count', 'Overdue_Visits_Count', 'Is_Anomaly', 'Anomaly_Score']]
df_with_anomalies.to_csv('outputs/anomaly_results.csv', index=False)
print(f"💾 Anomaly results saved to: outputs/anomaly_results.csv")

# Demo - show most anomalous patients
print(f"\n🚨 TOP 10 MOST ANOMALOUS PATIENTS:")
print("=" * 80)

top_anomalies = df.nlargest(10, 'Anomaly_Score')
for idx, (i, patient) in enumerate(top_anomalies.iterrows(), 1):
    print(f"\n{idx}. Patient: {patient['Subject ID']} | Study: {patient['Study'][:30]}")
    print(f"   Risk: {patient['Smart_Risk_Noisy']} (DQI: {patient['Smart_DQI_Noisy']:.1f})")
    print(f"   Issues: SAE={patient['SAE_Pending_Count']:.0f}, Visits={patient['Overdue_Visits_Count']:.0f}, Pages={patient['Missing_Pages']:.0f}")
    print(f"   Anomaly Score: {patient['Anomaly_Score']:.3f} {'🚨 ANOMALY' if patient['Is_Anomaly'] == 1 else ''}")
    
    # Explain why it's anomalous
    reasons = []
    if patient['SAE_Pending_Count'] > 10:
        reasons.append(f"Extremely high SAE count ({patient['SAE_Pending_Count']:.0f})")
    if patient['High_SAE_No_Visits'] == 1:
        reasons.append("High SAE but no overdue visits (unusual)")
    if patient['DQI_Gap'] > 10:
        reasons.append(f"Large DQI gap ({patient['DQI_Gap']:.1f})")
    if patient['Smart_DQI_Noisy'] < 60:
        reasons.append(f"Extremely low DQI ({patient['Smart_DQI_Noisy']:.1f})")
    
    if reasons:
        print(f"   Why Anomalous: {'; '.join(reasons)}")

print("\n" + "=" * 80)
print("✅ MODEL 3: ANOMALY DETECTION COMPLETE!")
print(f"🔍 Detected {n_anomalies:,} anomalous patients ({pct_anomalies:.1f}%)")
print("=" * 80)
