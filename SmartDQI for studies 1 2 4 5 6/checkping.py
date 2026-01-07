import pandas as pd
import os

for study in ['Study_5', 'Study_6']:
    path = f'data/raw_studies/{study}'
    files = os.listdir(path)
    edc_file = [f for f in files if 'CPID' in f or 'EDC' in f][0]
    
    df = pd.read_excel(f'{path}/{edc_file}', sheet_name=1, header=1)
    
    print(f"\n{study}:")
    print(f"  Patients: {len(df)}")
    print(f"  Missing Visits: {df.get('# Missing Visits', pd.Series([0])).sum()}")
    print(f"  Missing Pages: {df.get('# Missing Pages', pd.Series([0])).sum()}")
    
    # Check SAE
    sae_file = [f for f in files if 'SAE' in f][0]
    df_sae = pd.read_excel(f'{path}/{sae_file}', sheet_name=0)
    print(f"  SAE Records: {len(df_sae)}")
