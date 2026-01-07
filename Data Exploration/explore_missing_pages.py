import pandas as pd

file_path = 'data/raw_studies/Study_1/Study 1_Missing_Pages_Report_URSV3.0_14 NOV 2025_updated.xlsx'

print("📂 Loading Missing Pages Report...")

df = pd.read_excel(file_path)

print(f"✅ Loaded: {len(df)} rows")
print(f"📊 Columns: {len(df.columns)}")

print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n🔍 First 3 rows:")
print(df.head(3))

df.to_csv('outputs/study1_missing_pages.csv', index=False)
print("\n✅ Saved to outputs/study1_missing_pages.csv")
