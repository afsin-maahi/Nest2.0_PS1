import pandas as pd

df = pd.read_csv('outputs/study1_final_dqi.csv')

print("🚨 BOTTOM 5 PATIENTS - DETAILED VIEW:")
print("=" * 80)

bottom_5 = df.nsmallest(5, 'DQI_Score')[['Subject ID', 'Site ID', 'DQI_Score', 'Risk_Level', 
                                          'Overdue_Visits_Count', 'Missing_Pages_Count', 'SAE_Pending_Count']]

print(bottom_5.to_string(index=False))

print("\n" + "=" * 80)
print("\n📊 Issue Breakdown for Worst Patient:")
worst = df.loc[df['DQI_Score'].idxmin()]
print(f"Subject: {worst['Subject ID']} at {worst['Site ID']}")
print(f"DQI Score: {worst['DQI_Score']}")
print(f"  - Overdue Visits: {worst['Overdue_Visits_Count']}")
print(f"  - Missing Pages: {worst['Missing_Pages_Count']}")
print(f"  - Pending SAE Reviews: {worst['SAE_Pending_Count']}")

# Calculate deductions
deductions = 0
deductions += min(worst['Overdue_Visits_Count'] * 10, 30)
deductions += min(worst['Missing_Pages_Count'] * 1, 30)
deductions += min(worst['SAE_Pending_Count'] * 10, 40)
print(f"\nScore Calculation: 100 - {deductions} = {worst['DQI_Score']}")
