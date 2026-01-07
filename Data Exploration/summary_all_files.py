import pandas as pd
import os

print("=" * 80)
print("📊 STUDY 1 - DATA LOADING SUMMARY")
print("=" * 80)

files = {
    '1. EDC Metrics (Main)': 'outputs/study1_cleaned.csv',
    '2. Visit Tracker': 'outputs/study1_visit_tracker.csv',
    '3. Missing Pages': 'outputs/study1_missing_pages.csv',
    '4. SAE Dashboard': 'outputs/study1_sae_dashboard.csv',
    '5. MedDRA Coding': 'outputs/study1_meddra_coding.csv',
    '6. WHODrug Coding': 'outputs/study1_whodrug_coding.csv',
    '7. Inactivated Forms': 'outputs/study1_inactivated_forms.csv',
    '8. Missing Labs': 'outputs/study1_missing_labs.csv',
    '9. EDRR Reconciliation': 'outputs/study1_edrr.csv'
}

for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"\n✅ {name}")
        print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    else:
        print(f"\n❌ {name} - NOT FOUND")

print("\n" + "=" * 80)
print("✅ ALL FILES LOADED!")
print("=" * 80)
print("\n🎯 READY FOR STEP 1.4: Calculate Basic DQI")
print("\nType 'yes' when ready to move to DQI calculation!")
