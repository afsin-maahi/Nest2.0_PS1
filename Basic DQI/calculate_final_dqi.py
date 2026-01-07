import pandas as pd

print("📂 Loading enriched data...")

# Load enriched main data
df = pd.read_csv('outputs/study1_enriched.csv')

# Load SAE to aggregate by patient
df_sae = pd.read_csv('outputs/study1_sae_dashboard.csv')

# Count pending SAE reviews per patient
sae_pending = df_sae[df_sae['Review Status'] != 'Review Completed'].groupby('Patient ID').size().reset_index(name='SAE_Pending_Count')

# Merge SAE into main
df = df.merge(sae_pending, left_on='Subject ID', right_on='Patient ID', how='left')
df['SAE_Pending_Count'] = df['SAE_Pending_Count'].fillna(0)

print(f"✅ Loaded {len(df)} patients with aggregated metrics\n")

# Calculate DQI
def calculate_dqi(row):
    """Calculate DQI based on real data"""
    
    overdue_visits = row['Overdue_Visits_Count']
    missing_pages = row['Missing_Pages_Count']
    sae_pending = row['SAE_Pending_Count']
    
    # Start at 100
    score = 100.0
    
    # Deduct for overdue visits (10 points each, max 30)
    score -= min(overdue_visits * 10, 30)
    
    # Deduct for missing pages (1 point each, max 30)
    score -= min(missing_pages * 1, 30)
    
    # Deduct for pending SAE (10 points each, max 40)
    score -= min(sae_pending * 10, 40)
    
    score = max(score, 0)
    
    # Risk level
    if score >= 85:
        risk = 'Low'
    elif score >= 70:
        risk = 'Medium'
    else:
        risk = 'High'
    
    # Clean status
    is_clean = (overdue_visits == 0 and missing_pages == 0 and sae_pending == 0)
    
    return pd.Series({
        'DQI_Score': round(score, 1),
        'Risk_Level': risk,
        'Is_Clean': is_clean
    })

# Apply DQI calculation
df[['DQI_Score', 'Risk_Level', 'Is_Clean']] = df.apply(calculate_dqi, axis=1)

print("=" * 80)
print("📊 FINAL DQI CALCULATION")
print("=" * 80)
print(f"\nTotal Patients: {len(df)}")
print(f"Average DQI: {df['DQI_Score'].mean():.1f}")
print(f"Clean Patients: {df['Is_Clean'].sum()} ({df['Is_Clean'].mean()*100:.1f}%)")

print("\n📊 Risk Distribution:")
print(df['Risk_Level'].value_counts())

print("\n🏆 Top 5 Patients (Highest DQI):")
print(df.nlargest(5, 'DQI_Score')[['Subject ID', 'Site ID', 'DQI_Score', 'Risk_Level']])

print("\n🚨 Bottom 5 Patients (Lowest DQI):")
print(df.nsmallest(5, 'DQI_Score')[['Subject ID', 'Site ID', 'DQI_Score', 'Overdue_Visits_Count', 'Missing_Pages_Count', 'SAE_Pending_Count']])

# Save final results
df.to_csv('outputs/study1_final_dqi.csv', index=False)
print("\n💾 Saved to: outputs/study1_final_dqi.csv")
print("\n✅ BASIC DQI COMPLETE! Ready for visualization.")
