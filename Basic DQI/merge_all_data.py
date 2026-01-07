import pandas as pd

print("📂 Loading all datasets...")

# Main patient data
df_main = pd.read_csv('outputs/study1_cleaned.csv')

# Supporting data
df_visits = pd.read_csv('outputs/study1_visit_tracker.csv')
df_missing_pages = pd.read_csv('outputs/study1_missing_pages.csv')
df_sae = pd.read_csv('outputs/study1_sae_dashboard.csv')

print(f"✅ Main: {len(df_main)} patients")
print(f"✅ Overdue visits: {len(df_visits)} records")
print(f"✅ Missing pages: {len(df_missing_pages)} records")
print(f"✅ SAE events: {len(df_sae)} records")

# Aggregate visits by Subject
visits_by_subject = df_visits.groupby('Subject').size().reset_index(name='Overdue_Visits_Count')

# Aggregate missing pages by Subject
pages_by_subject = df_missing_pages.groupby('Subject Name').size().reset_index(name='Missing_Pages_Count')

# Aggregate SAE by Patient (check column names first)
print("\n📋 SAE Dashboard columns:")
print(df_sae.columns.tolist())

print("\n🔍 Merging data...")

# Merge visits
df_main = df_main.merge(visits_by_subject, 
                        left_on='Subject ID', 
                        right_on='Subject', 
                        how='left')
df_main['Overdue_Visits_Count'] = df_main['Overdue_Visits_Count'].fillna(0)

# Merge missing pages
df_main = df_main.merge(pages_by_subject, 
                        left_on='Subject ID', 
                        right_on='Subject Name', 
                        how='left')
df_main['Missing_Pages_Count'] = df_main['Missing_Pages_Count'].fillna(0)

print(f"\n✅ Merged data: {len(df_main)} patients")
print(f"📊 Patients with overdue visits: {(df_main['Overdue_Visits_Count'] > 0).sum()}")
print(f"📊 Patients with missing pages: {(df_main['Missing_Pages_Count'] > 0).sum()}")

# Save enriched dataset
df_main.to_csv('outputs/study1_enriched.csv', index=False)
print(f"\n💾 Saved to: outputs/study1_enriched.csv")
print("\n✅ STEP 1.5 COMPLETE! Now we can calculate proper DQI.")
