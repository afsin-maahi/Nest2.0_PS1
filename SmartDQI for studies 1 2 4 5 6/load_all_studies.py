# Save this as: load_all_studies_fixed.py

import pandas as pd
import os
from pathlib import Path

def load_study_data(study_name, study_folder):
    """
    Load all 9 data files for a given study with flexible file matching
    """
    print(f"\n{'='*80}")
    print(f"📂 LOADING {study_name}")
    print(f"{'='*80}")
    
    base_path = f'data/raw_studies/{study_folder}'
    files = os.listdir(base_path)
    
    data = {}
    
    # 1. EDC Metrics
    try:
        edc_file = [f for f in files if 'CPID' in f or 'EDC' in f or 'Metrics' in f][0]
        print(f"✓ Loading {edc_file}")
        data['main'] = pd.read_excel(f'{base_path}/{edc_file}', sheet_name=1, header=1)
        print(f"  → {len(data['main'])} patients loaded")
    except Exception as e:
        print(f"✗ EDC Metrics: {str(e)[:50]}")
    
    # 2. Visit Tracker
    try:
        visit_file = [f for f in files if 'Visit' in f and ('Projection' in f or 'Tracker' in f)][0]
        data['visits'] = pd.read_excel(f'{base_path}/{visit_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['visits'])} visit records")
    except Exception as e:
        print(f"✗ Visit Tracker: {str(e)[:50]}")
        data['visits'] = pd.DataFrame()
    
    # 3. Missing Pages
    try:
        pages_file = [f for f in files if 'Missing' in f and 'Pages' in f][0]
        data['pages'] = pd.read_excel(f'{base_path}/{pages_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['pages'])} missing page records")
    except Exception as e:
        print(f"✗ Missing Pages: {str(e)[:50]}")
        data['pages'] = pd.DataFrame()
    
    # 4. SAE Dashboard
    try:
        sae_file = [f for f in files if 'SAE' in f][0]
        data['sae'] = pd.read_excel(f'{base_path}/{sae_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['sae'])} SAE records")
    except Exception as e:
        print(f"✗ SAE Dashboard: {str(e)[:50]}")
        data['sae'] = pd.DataFrame()
    
    # 5. Missing Labs
    try:
        labs_file = [f for f in files if 'Lab' in f and 'Missing' in f][0]
        data['labs'] = pd.read_excel(f'{base_path}/{labs_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['labs'])} lab records")
    except Exception as e:
        print(f"✗ Missing Labs: {str(e)[:50]}")
        data['labs'] = pd.DataFrame()
    
    # 6. EDRR
    try:
        edrr_file = [f for f in files if 'EDRR' in f or 'Compiled' in f][0]
        data['edrr'] = pd.read_excel(f'{base_path}/{edrr_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['edrr'])} EDRR records")
    except Exception as e:
        print(f"✗ EDRR: {str(e)[:50]}")
        data['edrr'] = pd.DataFrame()
    
    # 7. MedDRA Coding
    try:
        meddra_file = [f for f in files if 'MedDRA' in f or 'MedDra' in f or 'Medra' in f][0]
        data['meddra'] = pd.read_excel(f'{base_path}/{meddra_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['meddra'])} MedDRA records")
    except Exception as e:
        print(f"✗ MedDRA: {str(e)[:50]}")
        data['meddra'] = pd.DataFrame()
    
    # 8. WHODrug Coding
    try:
        who_file = [f for f in files if 'WHO' in f][0]
        data['who'] = pd.read_excel(f'{base_path}/{who_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['who'])} WHODrug records")
    except Exception as e:
        print(f"✗ WHODrug: {str(e)[:50]}")
        data['who'] = pd.DataFrame()
    
    # 9. Inactivated Forms
    try:
        inact_file = [f for f in files if 'Inactivated' in f][0]
        data['inactivated'] = pd.read_excel(f'{base_path}/{inact_file}', sheet_name=0, header=0)
        print(f"✓ Loaded {len(data['inactivated'])} inactivated records")
    except Exception as e:
        print(f"✗ Inactivated: {str(e)[:50]}")
        data['inactivated'] = pd.DataFrame()
    
    print(f"\n✅ {study_name} data loaded successfully!")
    return data

# Load 3 studies (including Study 4 for volume!)
print("🚀 BATCH LOADING 3 STUDIES")
print("="*80)

studies_config = {
    'Study_1': {'folder': 'Study_1', 'context': 'oncology_phase3'},
    'Study_2': {'folder': 'Study_2', 'context': 'cardiology_phase2'},
    'Study_4': {'folder': 'Study_4', 'context': 'respiratory_phase1'}
}

all_studies = {}

for study_name, config in studies_config.items():
    all_studies[study_name] = load_study_data(study_name, config['folder'])
    all_studies[study_name]['context'] = config['context']

print("\n" + "="*80)
print("📊 LOADING SUMMARY:")
print("="*80)

total = 0
for study_name, data in all_studies.items():
    if 'main' in data and len(data['main']) > 0:
        count = len(data['main'])
        total += count
        print(f"✅ {study_name}: {count} patients - Context: {data['context']}")

print(f"\n🎉 Total patients across 3 studies: {total}")
print("="*80)
