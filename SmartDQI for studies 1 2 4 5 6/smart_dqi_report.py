import pandas as pd

df = pd.read_csv('outputs/study1_smart_dqi_comparison.csv')

print("=" * 80)
print("📊 SMART DQI vs BASIC DQI - EXECUTIVE SUMMARY")
print("=" * 80)

print(f"\n🎯 Context: ONCOLOGY Phase 3, Late Enrollment")
print(f"   Safety Weight: 32% (vs 10% in basic)")
print(f"   Rationale: Cancer trials require maximum safety vigilance\n")

print("📈 Overall Statistics:")
print(f"   Average Basic DQI: {df['Basic_DQI'].mean():.1f}")
print(f"   Average Smart DQI: {df['Smart_DQI'].mean():.1f}")
print(f"   Correlation: {df['Basic_DQI'].corr(df['Smart_DQI']):.3f}")

# Find patients where classification changed
risk_changed = df[df['Basic_Risk'] != df['Smart_Risk']].copy()
print(f"\n🔄 Risk Classification Changes: {len(risk_changed)} patients")

if len(risk_changed) > 0:
    print("\n   Patients reclassified:")
    for _, row in risk_changed.head(10).iterrows():
        arrow = "↑" if row['Smart_Risk'] == 'Low' else "↓"
        print(f"   {arrow} {row['Subject_ID']:12s}: {row['Basic_Risk']:8s} → {row['Smart_Risk']:8s} (SAE: {row['SAE_Pending']:.0f})")

# Biggest revelations
print("\n💡 KEY INSIGHTS:")

high_sae_patients = df[df['SAE_Pending'] >= 5].copy()
if len(high_sae_patients) > 0:
    print(f"\n1. HIGH SAE BURDEN ({len(high_sae_patients)} patients):")
    print(f"   Smart DQI properly weights safety issues in oncology context")
    for _, row in high_sae_patients.nsmallest(3, 'Smart_DQI').iterrows():
        print(f"   🚨 {row['Subject_ID']}: {row['SAE_Pending']:.0f} pending SAE reviews")
        print(f"      Basic: {row['Basic_DQI']:.1f} | Smart: {row['Smart_DQI']:.1f} | Impact: {row['Difference']:.1f} pts")

clean_patients = df[(df['SAE_Pending'] == 0) & (df['Missing_Pages'] == 0) & (df['Overdue_Visits'] == 0)]
print(f"\n2. CLEAN PATIENTS: {len(clean_patients)} patients ({len(clean_patients)/len(df)*100:.1f}%)")
print(f"   These patients are ready for database lock")

print("\n3. ACTIONABLE PRIORITIES (Smart DQI ranking):")
urgent = df.nsmallest(5, 'Smart_DQI')
for i, (_, row) in enumerate(urgent.iterrows(), 1):
    print(f"   {i}. {row['Subject_ID']} (Site {row['Site_ID']})")
    print(f"      Smart DQI: {row['Smart_DQI']:.1f} | Focus: {row['SAE_Pending']:.0f} SAE + {row['Missing_Pages']:.0f} pages")

print("\n" + "=" * 80)
print("✅ Smart DQI provides context-aware prioritization")
print("   → Sites know where to focus efforts based on trial context")
print("=" * 80)
