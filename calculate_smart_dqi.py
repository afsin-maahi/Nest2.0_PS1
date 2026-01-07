import pandas as pd
from smart_dqi_contexts import get_smart_weights, BASE_WEIGHTS


print("📂 Loading Study 1 data with Basic DQI...")
df = pd.read_csv('outputs/study1_final_dqi.csv')


print(f"✅ Loaded {len(df)} patients\n")


# Define context for Study 1 (assume oncology phase 3, late enrollment)
study1_context = {
    'therapeutic_area': 'oncology',
    'phase': 'phase3',
    'timeline': 'late'
}


def calculate_smart_dqi(row, context):
    """
    Calculate Smart DQI with context-aware weights and AGGRESSIVE penalties
    """
    # Get context-aware weights
    smart_weights = get_smart_weights(
        context['therapeutic_area'],
        context['phase'],
        context['timeline']
    )
    
    # Calculate component scores (0-100 each)
    components = {}
    
    # Visits component - MORE AGGRESSIVE PENALTY
    overdue_visits = row['Overdue_Visits_Count']
    if overdue_visits == 0:
        components['visits'] = 100
    else:
        components['visits'] = max(0, 100 - (overdue_visits * 20))  # Increased from 15 to 20
    
    # Pages component - MORE AGGRESSIVE PENALTY
    missing_pages = row['Missing_Pages_Count']
    if missing_pages == 0:
        components['pages'] = 100
    else:
        components['pages'] = max(0, 100 - (missing_pages * 3))  # Increased from 2 to 3
    
    # Safety component (MOST CRITICAL) - EVEN HEAVIER PENALTY FOR ONCOLOGY
    sae_pending = row['SAE_Pending_Count']
    if sae_pending == 0:
        components['safety'] = 100
    else:
        # For oncology context, penalize SAE VERY heavily
        penalty = 25 if context['therapeutic_area'] == 'oncology' else 15
        components['safety'] = max(0, 100 - (sae_pending * penalty))  # 25 pts per SAE in oncology
    
    # Reduce baseline scores for queries/accuracy/verification to create more variation
    components['queries'] = 85      # Reduced from 90 to 85
    components['accuracy'] = 85     # Reduced from 90 to 85
    components['verification'] = 80 # Reduced from 90 to 80
    
    # Calculate Smart DQI
    smart_dqi = sum(components[key] * smart_weights[key] for key in components)
    
    # Calculate Basic DQI for comparison
    basic_dqi = sum(components[key] * BASE_WEIGHTS[key] for key in components)
    
    return {
        'Smart_DQI': round(smart_dqi, 1),
        'Basic_DQI': round(basic_dqi, 1),
        'Difference': round(smart_dqi - basic_dqi, 1),
        'Components': components,
        'Smart_Weights': smart_weights
    }


print(f"📋 Study 1 Context: {study1_context['therapeutic_area'].upper()} - {study1_context['phase'].upper()} - {study1_context['timeline'].upper()} enrollment")
print("=" * 80)


# Calculate Smart DQI for all patients
results = []
for idx, row in df.iterrows():
    smart_result = calculate_smart_dqi(row, study1_context)
    results.append({
        'Subject_ID': row['Subject ID'],
        'Site_ID': row['Site ID'],
        'Basic_DQI': row['DQI_Score'],
        'Smart_DQI': smart_result['Smart_DQI'],
        'Difference': smart_result['Difference'],
        'Overdue_Visits': row['Overdue_Visits_Count'],
        'Missing_Pages': row['Missing_Pages_Count'],
        'SAE_Pending': row['SAE_Pending_Count']
    })


results_df = pd.DataFrame(results)


# Determine Smart Risk Level - MUCH STRICTER THRESHOLDS
def get_risk_level(score):
    if score >= 92:  # Changed from 90 to 92
        return 'Low'
    elif score >= 80:  # Changed from 75 to 80
        return 'Medium'
    else:
        return 'High'


results_df['Smart_Risk'] = results_df['Smart_DQI'].apply(get_risk_level)
results_df['Basic_Risk'] = results_df['Basic_DQI'].apply(get_risk_level)


print("\n📊 SMART DQI RESULTS:")
print("=" * 80)
print(f"Average Basic DQI: {results_df['Basic_DQI'].mean():.1f}")
print(f"Average Smart DQI: {results_df['Smart_DQI'].mean():.1f}")
print(f"Average Difference: {results_df['Difference'].mean():.1f} points")


print("\n📊 Risk Distribution:")
print("Basic DQI Risk:")
basic_risk_counts = results_df['Basic_Risk'].value_counts()
for risk in ['High', 'Medium', 'Low']:
    if risk in basic_risk_counts.index:
        count = basic_risk_counts[risk]
        pct = count / len(results_df) * 100
        print(f"  {risk:8s}: {count:3d} ({pct:5.1f}%)")

print("\nSmart DQI Risk:")
smart_risk_counts = results_df['Smart_Risk'].value_counts()
for risk in ['High', 'Medium', 'Low']:
    if risk in smart_risk_counts.index:
        count = smart_risk_counts[risk]
        pct = count / len(results_df) * 100
        print(f"  {risk:8s}: {count:3d} ({pct:5.1f}%)")


print("\n📊 Risk Classification Changes:")
risk_changes = results_df[results_df['Basic_Risk'] != results_df['Smart_Risk']]
print(f"Patients with changed risk level: {len(risk_changes)} ({len(risk_changes)/len(results_df)*100:.1f}%)")


print("\n🚨 BIGGEST IMPACT - Patients where Smart DQI reveals hidden risk:")
high_impact = results_df[results_df['Difference'] < -10].nlargest(5, 'SAE_Pending')
if len(high_impact) > 0:
    print(high_impact[['Subject_ID', 'Site_ID', 'Basic_DQI', 'Smart_DQI', 'Difference', 'SAE_Pending']].to_string(index=False))
else:
    print("No patients with significant negative difference found.")


print("\n🏆 Top 5 by Smart DQI:")
print(results_df.nlargest(5, 'Smart_DQI')[['Subject_ID', 'Site_ID', 'Smart_DQI', 'Basic_DQI']].to_string(index=False))


print("\n🚨 Bottom 5 by Smart DQI:")
print(results_df.nsmallest(5, 'Smart_DQI')[['Subject_ID', 'Site_ID', 'Smart_DQI', 'Basic_DQI', 'SAE_Pending']].to_string(index=False))


# Save
results_df.to_csv('outputs/study1_smart_dqi_comparison.csv', index=False)
print("\n💾 Saved to: outputs/study1_smart_dqi_comparison.csv")
print("\n✅ SMART DQI COMPLETE!")
