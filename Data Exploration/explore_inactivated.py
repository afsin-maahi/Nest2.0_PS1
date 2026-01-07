import pandas as pd

file_path = 'data/raw_studies/Study_1/Study 1_Inactivated Forms, Folders and  Records Report_updated.xlsx'

print("📂 Loading Inactivated Forms Report...")

df = pd.read_excel(file_path)

print(f"✅ Loaded: {len(df)} rows")
print(f"📊 Columns: {len(df.columns)}")

print("\n📋 COLUMN NAMES:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\n🔍 First 3 rows:")
print(df.head(3))

df.to_csv('outputs/study1_inactivated_forms.csv', index=False)
print("\n✅ Saved to outputs/study1_inactivated_forms.csv")
