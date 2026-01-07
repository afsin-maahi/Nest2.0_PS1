import pandas as pd
import os

file_path = 'data/raw_studies/Study_1/Study_1 CPID_EDC_Metrics_URSV2.0_14 NOV 2025_updated.xlsx'

print("📂 Loading Study 1 EDC Metrics...")

df_raw = pd.read_excel(file_path, sheet_name='Subject Level Metrics', header=None)

# Row 0 = general categories, Row 1 = specific metrics
header_row0 = df_raw.iloc[0].fillna('')
header_row1 = df_raw.iloc[1].fillna('')

# PREFER row1 (more specific), fallback to row0
final_headers = []
for h0, h1 in zip(header_row0, header_row1):
    h1_str = str(h1).strip()
    h0_str = str(h0).strip()
    
    if h1_str and h1_str != 'nan':
        final_headers.append(h1_str)
    elif h0_str and h0_str != 'nan':
        final_headers.append(h0_str)
    else:
        final_headers.append(f'Unnamed_{len(final_headers)}')

# Data starts at row 3 (skip: 0=header, 1=header, 2=responsible)
df = df_raw.iloc[3:].copy()
df.columns = final_headers
df = df.reset_index(drop=True)

# Remove any remaining header rows
df = df[df['Project Name'] == 'Study 1'].copy()
df = df.reset_index(drop=True)

print(f"\n✅ Successfully loaded!")
print(f"📊 Total Patients: {len(df)}")

print("\n🔍 First 3 patients:")
print(df.head(3)[['Subject ID', 'Site ID', 'Country', 'Missing Visits', 'Missing Page']])

print("\n📋 KEY DQI COLUMNS:")
dqi_cols = ['Subject ID', 'Site ID', 'Missing Visits', 'Missing Page', 
            '# eSAE dashboard review for DM', '# Safety Queries']
for col in dqi_cols:
    status = "✅" if col in df.columns else "❌"
    print(f"{status} {col}")

df.to_csv('outputs/study1_cleaned.csv', index=False)
print(f"\n💾 Saved to: outputs/study1_cleaned.csv")
print(f"✅ STEP 1.3 COMPLETE!")
