import pandas as pd

file_path = 'data/raw_studies/Study_1/Study 1_eSAE Dashboard_Standard DM_Safety Report_updated.xlsx'

print("📂 Loading SAE Dashboard...")

# This file likely has multiple sheets
xls = pd.ExcelFile(file_path)
print(f"📋 Available sheets: {xls.sheet_names}")

# Load first sheet
df = pd.read_excel(file_path, sheet_name=0)

print(f"\n✅ Loaded sheet 1: {len(df)} rows")
print(f"📊 Columns: {len(df.columns)}")

print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n🔍 First 3 rows:")
print(df.head(3))

df.to_csv('outputs/study1_sae_dashboard.csv', index=False)
print("\n✅ Saved to outputs/study1_sae_dashboard.csv")
