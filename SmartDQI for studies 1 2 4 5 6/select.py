import os
import pandas as pd

studies = ['Study_1', 'Study_2', 'Study_4', 'Study_5', 'Study_6']

print("📊 STUDY INVENTORY:")
print("=" * 80)

for study in studies:
    path = f'data/raw_studies/{study}'
    files = os.listdir(path)
    
    # Try to load the main metrics file to count patients
    try:
        metrics_file = [f for f in files if 'EDC' in f or 'Metrics' in f][0]
        df = pd.read_excel(f'{path}/{metrics_file}', sheet_name=1)
        patient_count = len(df)
        print(f"\n✅ {study}:")
        print(f"   Files: {len(files)}")
        print(f"   Patients: {patient_count}")
        print(f"   Key file: {metrics_file}")
    except Exception as e:
        print(f"\n⚠️  {study}:")
        print(f"   Files: {len(files)}")
        print(f"   Error reading: {str(e)[:50]}")

print("\n" + "=" * 80)
