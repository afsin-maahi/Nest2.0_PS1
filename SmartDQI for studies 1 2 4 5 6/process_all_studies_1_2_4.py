import pandas as pd
from aggregate_study_fixed import aggregate_study
from smart_dqi_contexts import get_smart_weights, BASE_WEIGHTS

print("="*80)
print("🚀 MULTI-STUDY SMART DQI ANALYSIS")
print("="*80)


studies_config = [
    {'folder': 'Study_1', 'name': 'Study_1', 'therapeutic_area': 'oncology', 'phase': 'phase3'},
    {'folder': 'Study_2', 'name': 'Study_2', 'therapeutic_area': 'cardiology', 'phase': 'phase2'},
    {'folder': 'Study_4', 'name': 'Study_4', 'therapeutic_area': 'oncology', 'phase': 'phase3'},
    {'folder': 'Study_5', 'name': 'Study_5', 'therapeutic_area': 'cardiology', 'phase': 'phase2'},
    {'folder': 'Study_6', 'name': 'Study_6', 'therapeutic_area': 'oncology', 'phase': 'phase2'}  # Changed to phase2
]


def calculate_dqi(row, therapeutic_area, phase):
    """Calculate Smart DQI and Basic DQI"""
    smart_weights = get_smart_weights(therapeutic_area, phase, 'late')
    
    # Component scores
    visits = float(row['Overdue_Visits_Count'])
    pages = float(row['Missing_Pages'])
    sae = float(row['SAE_Pending_Count'])
    
    visits_score = 100 if visits == 0 else max(0, 100 - (visits * 10))
    pages_score = 100 if pages == 0 else max(0, 100 - pages)
    safety_score = 100 if sae == 0 else max(0, 100 - (sae * 10))
    
    components = {
        'visits': visits_score,
        'pages': pages_score,
        'safety': safety_score,
        'queries': 90,
        'accuracy': 90,
        'verification': 90
    }
    
    smart_dqi = sum(components[key] * smart_weights[key] for key in components)
    basic_dqi = sum(components[key] * BASE_WEIGHTS[key] for key in components)
    
    return smart_dqi, basic_dqi

# Process all studies
all_results = []

for config in studies_config:
    # Aggregate data
    df_agg = aggregate_study(config['folder'], config['name'])
    df_agg['Context'] = f"{config['therapeutic_area']}_{config['phase']}"
    
    # Calculate DQI for each patient
    dqi_scores = []
    for _, row in df_agg.iterrows():
        smart_dqi, basic_dqi = calculate_dqi(row, config['therapeutic_area'], config['phase'])
        dqi_scores.append({
            'Smart_DQI': round(smart_dqi, 1),
            'Basic_DQI': round(basic_dqi, 1),
            'Difference': round(smart_dqi - basic_dqi, 1)
        })
    
    df_dqi = pd.DataFrame(dqi_scores)
    df_final = pd.concat([df_agg.reset_index(drop=True), df_dqi], axis=1)
    
    all_results.append(df_final)
    
    print(f"\n✅ {config['name']} processed: {len(df_final)} patients")
    print(f"   → Context: {config['therapeutic_area']} {config['phase']}")
    print(f"   → Avg Smart DQI: {df_final['Smart_DQI'].mean():.1f}")

# Combine all studies
final_df = pd.concat(all_results, ignore_index=True)

# Add risk levels
def get_risk_level(score):
    if score >= 85:
        return 'Low'
    elif score >= 70:
        return 'Medium'
    else:
        return 'High'

final_df['Smart_Risk'] = final_df['Smart_DQI'].apply(get_risk_level)
final_df['Basic_Risk'] = final_df['Basic_DQI'].apply(get_risk_level)

# Save
final_df.to_csv('outputs/all_studies_smart_dqi.csv', index=False)

print("\n" + "="*80)
print("📊 FINAL MULTI-STUDY SUMMARY")
print("="*80)

print(f"\n🎯 Total Patients: {len(final_df)}")

print(f"\n📋 By Study:")
for study in final_df['Study'].unique():
    sdf = final_df[final_df['Study'] == study]
    context = sdf['Context'].iloc[0]
    print(f"\n  {study} ({context}):")
    print(f"    Patients: {len(sdf)}")
    print(f"    Avg Smart DQI: {sdf['Smart_DQI'].mean():.1f}")
    print(f"    Avg Basic DQI: {sdf['Basic_DQI'].mean():.1f}")
    print(f"    High Risk: {len(sdf[sdf['Smart_Risk'] == 'High'])}")
    print(f"    Medium Risk: {len(sdf[sdf['Smart_Risk'] == 'Medium'])}")
    print(f"    Low Risk: {len(sdf[sdf['Smart_Risk'] == 'Low'])}")

print(f"\n🎯 Overall Risk Distribution (Smart DQI):")
risk_dist = final_df['Smart_Risk'].value_counts()
for risk, count in risk_dist.items():
    pct = (count / len(final_df)) * 100
    print(f"  {risk:8s}: {count:4d} ({pct:5.1f}%)")

print(f"\n🚨 Top 10 High-Risk Patients (Smart DQI):")
top_risk = final_df.nsmallest(10, 'Smart_DQI')[['Study', 'Subject ID', 'Site ID', 'Smart_DQI', 'SAE_Pending_Count', 'Missing_Pages', 'Overdue_Visits_Count']]
print(top_risk.to_string(index=False))

print(f"\n🏆 Top 10 Best Performers (Smart DQI):")
top_performers = final_df.nlargest(10, 'Smart_DQI')[['Study', 'Subject ID', 'Site ID', 'Smart_DQI']]
print(top_performers.to_string(index=False))

print(f"\n📊 Context Comparison:")
for context in final_df['Context'].unique():
    cdf = final_df[final_df['Context'] == context]
    print(f"\n  {context}:")
    print(f"    Patients: {len(cdf)}")
    print(f"    Avg Smart DQI: {cdf['Smart_DQI'].mean():.1f}")
    print(f"    High Risk %: {len(cdf[cdf['Smart_Risk'] == 'High']) / len(cdf) * 100:.1f}%")

print(f"\n💾 Saved to: outputs/all_studies_smart_dqi.csv")
print("="*80)
print("✅ MULTI-STUDY ANALYSIS COMPLETE!")
print("="*80)
