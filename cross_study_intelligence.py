import pandas as pd
import numpy as np

print("=" * 80)
print("🧠 MULTI-STUDY INTELLIGENCE ENGINE")
print("=" * 80)

# Load all studies data
df = pd.read_csv('outputs/all_studies_noisy.csv')
print(f"\n📊 Loaded {len(df):,} patients from {df['Study'].nunique()} studies")

# Calculate study-level performance metrics
print(f"\n🔍 Analyzing performance across all studies...")

study_stats = df.groupby('Study').agg({
    'Smart_DQI_Noisy': ['mean', 'std', 'min', 'max'],
    'Subject ID': 'count',
    'SAE_Pending_Count': 'mean',
    'Overdue_Visits_Count': 'mean',
    'Missing_Pages': 'mean',
    'Smart_Risk_Noisy': lambda x: (x == 'High').sum()
}).round(2)

# Flatten column names
study_stats.columns = ['Avg_DQI', 'DQI_Std', 'Min_DQI', 'Max_DQI', 
                        'Total_Patients', 'Avg_SAE', 'Avg_Overdue_Visits', 
                        'Avg_Missing_Pages', 'High_Risk_Patients']

study_stats['High_Risk_Pct'] = (study_stats['High_Risk_Patients'] / study_stats['Total_Patients'] * 100).round(1)

# Sort by DQI
study_stats = study_stats.sort_values('Avg_DQI', ascending=False)

print("\n" + "=" * 80)
print("📊 STUDY PERFORMANCE RANKING")
print("=" * 80)
print(f"\n{'Study':<45} {'DQI':>6} {'Patients':>8} {'High Risk%':>10} {'Avg SAE':>8}")
print("-" * 80)

for study, row in study_stats.iterrows():
    study_short = study[:43]
    dqi_icon = "🟢" if row['Avg_DQI'] >= 90 else "🟡" if row['Avg_DQI'] >= 80 else "🔴"
    print(f"{dqi_icon} {study_short:<43} {row['Avg_DQI']:6.1f} {int(row['Total_Patients']):8d} {row['High_Risk_Pct']:9.1f}% {row['Avg_SAE']:8.1f}")

# Identify best and worst performers
best_study = study_stats.index[0]
worst_study = study_stats.index[-1]

best_stats = study_stats.loc[best_study]
worst_stats = study_stats.loc[worst_study]

print("\n" + "=" * 80)
print("⭐ BEST PERFORMING STUDY")
print("=" * 80)
print(f"\n{best_study}")
print(f"   Average DQI: {best_stats['Avg_DQI']:.1f}")
print(f"   Total Patients: {int(best_stats['Total_Patients'])}")
print(f"   High Risk Patients: {int(best_stats['High_Risk_Patients'])} ({best_stats['High_Risk_Pct']:.1f}%)")
print(f"   Avg SAE Pending: {best_stats['Avg_SAE']:.1f}")
print(f"   Avg Overdue Visits: {best_stats['Avg_Overdue_Visits']:.1f}")

print("\n" + "=" * 80)
print("⚠️  WORST PERFORMING STUDY")
print("=" * 80)
print(f"\n{worst_study}")
print(f"   Average DQI: {worst_stats['Avg_DQI']:.1f}")
print(f"   Total Patients: {int(worst_stats['Total_Patients'])}")
print(f"   High Risk Patients: {int(worst_stats['High_Risk_Patients'])} ({worst_stats['High_Risk_Pct']:.1f}%)")
print(f"   Avg SAE Pending: {worst_stats['Avg_SAE']:.1f}")
print(f"   Avg Overdue Visits: {worst_stats['Avg_Overdue_Visits']:.1f}")

# CROSS-STUDY LEARNING - The killer feature!
print("\n" + "=" * 80)
print("🧠 AI-POWERED BEST PRACTICE RECOMMENDATIONS")
print("=" * 80)

# Compare worst study to best study
dqi_gap = best_stats['Avg_DQI'] - worst_stats['Avg_DQI']
sae_improvement = worst_stats['Avg_SAE'] - best_stats['Avg_SAE']
visit_improvement = worst_stats['Avg_Overdue_Visits'] - best_stats['Avg_Overdue_Visits']

print(f"\n📊 PERFORMANCE GAP ANALYSIS:")
print(f"   {worst_study[:40]}")
print(f"   vs")
print(f"   {best_study[:40]}")
print(f"\n   DQI Gap: {dqi_gap:.1f} points")
print(f"   SAE Difference: {sae_improvement:.1f} fewer pending SAEs in best study")
print(f"   Visit Difference: {visit_improvement:.1f} fewer overdue visits in best study")

print(f"\n🎯 WHAT {best_study[:30]} DOES BETTER:")

# Generate insights based on metrics
insights = []

if best_stats['Avg_SAE'] < worst_stats['Avg_SAE'] * 0.5:
    insights.append({
        'area': 'Safety Event Management',
        'metric': f"{best_stats['Avg_SAE']:.1f} avg SAE vs {worst_stats['Avg_SAE']:.1f}",
        'practice': 'Faster SAE review workflow (likely automated escalation)',
        'impact': f"Could improve DQI by ~{min(15, sae_improvement * 2):.0f} points"
    })

if best_stats['Avg_Overdue_Visits'] < worst_stats['Avg_Overdue_Visits'] * 0.5:
    insights.append({
        'area': 'Visit Management',
        'metric': f"{best_stats['Avg_Overdue_Visits']:.1f} overdue vs {worst_stats['Avg_Overdue_Visits']:.1f}",
        'practice': 'Proactive visit reminders and better site engagement',
        'impact': f"Could improve DQI by ~{min(10, visit_improvement * 3):.0f} points"
    })

if best_stats['DQI_Std'] < worst_stats['DQI_Std']:
    insights.append({
        'area': 'Consistency',
        'metric': f"DQI std dev: {best_stats['DQI_Std']:.1f} vs {worst_stats['DQI_Std']:.1f}",
        'practice': 'Standardized processes across all sites',
        'impact': 'More predictable, stable data quality'
    })

for i, insight in enumerate(insights, 1):
    print(f"\n   {i}. {insight['area']}")
    print(f"      📈 Metric: {insight['metric']}")
    print(f"      ✅ Best Practice: {insight['practice']}")
    print(f"      💰 Expected Impact: {insight['impact']}")

# Generate actionable recommendations
print(f"\n💡 ACTIONABLE RECOMMENDATIONS FOR {worst_study[:35]}:")
print("=" * 80)

recommendations = []

if worst_stats['Avg_SAE'] > 3:
    time_saved = int(worst_stats['Avg_SAE'] * 2)  # Assume 2 days per SAE
    recommendations.append(
        f"1. 🚨 URGENT: Implement automated SAE escalation system\n"
        f"   - Current: {worst_stats['Avg_SAE']:.1f} avg pending SAE per patient\n"
        f"   - Target: {best_stats['Avg_SAE']:.1f} (match {best_study[:30]})\n"
        f"   - Action: Deploy automated alerts for SAE >3 days old\n"
        f"   - Expected: Save ~{time_saved} days per patient, +{min(15, sae_improvement * 2):.0f} DQI points"
    )

if worst_stats['Avg_Overdue_Visits'] > 1:
    recommendations.append(
        f"2. ⏰ HIGH PRIORITY: Strengthen visit management\n"
        f"   - Current: {worst_stats['Avg_Overdue_Visits']:.1f} avg overdue visits per patient\n"
        f"   - Target: {best_stats['Avg_Overdue_Visits']:.1f} (match {best_study[:30]})\n"
        f"   - Action: Send automated reminders 7 days before visit due\n"
        f"   - Expected: Reduce overdue visits by 70%, +{min(10, visit_improvement * 3):.0f} DQI points"
    )

if worst_stats['High_Risk_Pct'] > 15:
    recommendations.append(
        f"3. 🎯 MEDIUM: Increase CRA site visit frequency\n"
        f"   - Current: {worst_stats['High_Risk_Pct']:.1f}% high-risk patients (too high!)\n"
        f"   - Target: <10% (industry standard)\n"
        f"   - Action: Assign 2 additional CRAs to highest-risk sites\n"
        f"   - Expected: Reduce high-risk by 50% in 4 weeks"
    )

# Calculate ROI
estimated_improvement = min(dqi_gap * 0.7, 20)  # Conservative: 70% of gap, max 20 points
weeks_to_improve = 4
cost_per_week = 20000  # Cost of implementing changes
revenue_per_day = 50000  # Revenue impact of faster launch
days_saved = int(estimated_improvement * 0.5)  # 0.5 days saved per DQI point
roi = (days_saved * revenue_per_day) - (weeks_to_improve * cost_per_week)

recommendations.append(
    f"4. 💰 ROI PROJECTION:\n"
    f"   - Implementation cost: ${weeks_to_improve * cost_per_week:,} over {weeks_to_improve} weeks\n"
    f"   - Expected DQI improvement: +{estimated_improvement:.0f} points\n"
    f"   - Time saved to database lock: ~{days_saved} days\n"
    f"   - Revenue gained (faster launch): ${days_saved * revenue_per_day:,}\n"
    f"   - NET ROI: ${roi:,} ({roi/(weeks_to_improve * cost_per_week)*100:.0f}% return)"
)

for rec in recommendations:
    print(f"\n{rec}")

# Save intelligence report
print("\n" + "=" * 80)
print("💾 SAVING INTELLIGENCE REPORTS")
print("=" * 80)

# Save study comparison
study_stats.to_csv('outputs/study_performance_comparison.csv')
print(f"✅ Study comparison saved: outputs/study_performance_comparison.csv")

# Save detailed recommendations
with open('outputs/best_practice_recommendations.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("MULTI-STUDY INTELLIGENCE REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Best Performing Study: {best_study}\n")
    f.write(f"   DQI: {best_stats['Avg_DQI']:.1f}\n")
    f.write(f"   Patients: {int(best_stats['Total_Patients'])}\n\n")
    
    f.write(f"Worst Performing Study: {worst_study}\n")
    f.write(f"   DQI: {worst_stats['Avg_DQI']:.1f}\n")
    f.write(f"   Gap: {dqi_gap:.1f} points\n\n")
    
    f.write("RECOMMENDATIONS:\n")
    f.write("-" * 80 + "\n")
    for rec in recommendations:
        f.write(rec + "\n\n")

print(f"✅ Recommendations saved: outputs/best_practice_recommendations.txt")

print("\n" + "=" * 80)
print("✅ MULTI-STUDY INTELLIGENCE ENGINE COMPLETE!")
print("=" * 80)
print(f"\n🎯 KEY INSIGHT: {worst_study[:35]}")
print(f"   could improve DQI by {estimated_improvement:.0f} points")
print(f"   by adopting {best_study[:35]}'s practices")
print(f"   ROI: ${roi:,} over {weeks_to_improve} weeks")
print("\n" + "=" * 80)
