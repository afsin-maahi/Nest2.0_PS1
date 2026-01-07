import pandas as pd
import os
import time
from smart_dqi_contexts import get_smart_weights, BASE_WEIGHTS


print("=" * 80)
print("🚀 LOADING ALL STUDIES WITH SMART DQI")
print("=" * 80)


# Auto-discover all study folders
base_path = 'data/raw_studies'
all_study_folders = sorted([d for d in os.listdir(base_path) if d.startswith('Study')])
print(f"📁 Found {len(all_study_folders)} studies\n")


# Define contexts (cycle through for variety)
context_patterns = [
    {'therapeutic_area': 'oncology', 'phase': 'phase3'},
    {'therapeutic_area': 'cardiology', 'phase': 'phase2'},
    {'therapeutic_area': 'oncology', 'phase': 'phase2'},
    {'therapeutic_area': 'cardiology', 'phase': 'phase3'},
]


def aggregate_study(study_folder, study_name, context):
    """
    Aggregate all data for a single study
    Returns DataFrame with Subject_ID, Site_ID, and all metrics
    """
    study_path = f'{base_path}/{study_folder}'
    files = os.listdir(study_path)
    
    # 1. Load main EDC metrics
    edc_files = [f for f in files if 'CPID' in f or 'EDC' in f]
    if not edc_files:
        raise Exception("No EDC file found")
    
    df_main = pd.read_excel(f'{study_path}/{edc_files[0]}', sheet_name=1, header=1)
    df_main.columns = df_main.columns.str.strip()
    
    # Check what columns exist
    subject_col = None
    site_col = None
    
    for col in df_main.columns:
        if 'Subject' in col and 'ID' in col:
            subject_col = col
        if 'Site' in col and 'ID' in col:
            site_col = col
    
    if not subject_col or not site_col:
        raise Exception(f"Missing Subject/Site ID columns. Found: {df_main.columns.tolist()[:5]}")
    
    # Rename to standardized names
    df_clean = df_main.rename(columns={
        subject_col: 'Subject_ID',
        site_col: 'Site_ID'
    })
    
    # Keep only rows where Subject_ID is valid
    df_clean = df_clean[df_clean['Subject_ID'].notna()].copy()
    df_clean['Subject_ID'] = df_clean['Subject_ID'].astype(str)
    
    # Initialize metrics
    df_clean['Overdue_Visits_Count'] = 0
    df_clean['Missing_Pages'] = 0
    df_clean['SAE_Pending_Count'] = 0
    
    # 2. Try to load Visit Tracker
    visit_files = [f for f in files if 'Visit' in f and ('Projection' in f or 'Tracker' in f)]
    if visit_files:
        try:
            df_visits = pd.read_excel(f'{study_path}/{visit_files[0]}', sheet_name=0, header=0)
            if len(df_visits) > 0:
                visit_col = [c for c in df_visits.columns if 'Subject' in c][0]
                df_visits[visit_col] = df_visits[visit_col].astype(str)
                visits_count = df_visits.groupby(visit_col).size().reset_index(name='Overdue_Visits_Count')
                visits_count.columns = ['Subject_ID', 'Overdue_Visits_Count']
                df_clean = df_clean.merge(visits_count, on='Subject_ID', how='left', suffixes=('', '_new'))
                if 'Overdue_Visits_Count_new' in df_clean.columns:
                    df_clean['Overdue_Visits_Count'] = df_clean['Overdue_Visits_Count_new'].fillna(0)
                    df_clean = df_clean.drop(columns=['Overdue_Visits_Count_new'])
        except Exception as e:
            pass
    
    # 3. Try to load SAE Dashboard
    sae_files = [f for f in files if 'SAE' in f]
    if sae_files:
        try:
            df_sae = pd.read_excel(f'{study_path}/{sae_files[0]}', sheet_name=0, header=0)
            if len(df_sae) > 0 and len(df_sae.columns) > 3:
                sae_subj_col = df_sae.columns[3]
                df_sae[sae_subj_col] = df_sae[sae_subj_col].astype(str)
                status_cols = [c for c in df_sae.columns if 'Review' in str(c) and 'Status' in str(c)]
                if status_cols:
                    status_col = status_cols[0]
                    df_sae_pending = df_sae[df_sae[status_col].astype(str).str.contains('Pending|Review', case=False, na=False)]
                    sae_count = df_sae_pending.groupby(sae_subj_col).size().reset_index(name='SAE_Pending_Count')
                    sae_count.columns = ['Subject_ID', 'SAE_Pending_Count']
                    df_clean = df_clean.merge(sae_count, on='Subject_ID', how='left')
                    if 'SAE_Pending_Count_y' in df_clean.columns:
                        df_clean['SAE_Pending_Count'] = df_clean['SAE_Pending_Count_y'].fillna(0)
                        df_clean = df_clean.drop(columns=['SAE_Pending_Count_x', 'SAE_Pending_Count_y'], errors='ignore')
        except Exception as e:
            pass
    
    # Fill missing columns
    df_clean['Overdue_Visits_Count'] = pd.to_numeric(df_clean['Overdue_Visits_Count'], errors='coerce').fillna(0)
    df_clean['Missing_Pages'] = pd.to_numeric(df_clean.get('Missing_Pages', 0), errors='coerce').fillna(0)
    df_clean['SAE_Pending_Count'] = pd.to_numeric(df_clean['SAE_Pending_Count'], errors='coerce').fillna(0)
    
    # Add study info
    df_clean['Study'] = study_name
    df_clean['Context'] = f"{context['therapeutic_area']}_{context['phase']}"
    
    # Select final columns
    final_cols = ['Subject_ID', 'Site_ID', 'Study', 'Context', 'Overdue_Visits_Count', 'Missing_Pages', 'SAE_Pending_Count']
    return df_clean[final_cols].copy()


def calculate_dqi(row, therapeutic_area, phase):
    """
    Calculate Smart DQI with AGGRESSIVE penalties
    """
    smart_weights = get_smart_weights(therapeutic_area, phase, 'late')
    
    visits = float(row['Overdue_Visits_Count'])
    pages = float(row['Missing_Pages'])
    sae = float(row['SAE_Pending_Count'])
    
    # Component scores with AGGRESSIVE penalties
    visits_score = 100 if visits == 0 else max(0, 100 - (visits * 20))
    pages_score = 100 if pages == 0 else max(0, 100 - (pages * 3))
    
    # Oncology gets VERY heavy SAE penalty
    penalty = 25 if therapeutic_area == 'oncology' else 15
    safety_score = 100 if sae == 0 else max(0, 100 - (sae * penalty))
    
    components = {
        'visits': visits_score,
        'pages': pages_score,
        'safety': safety_score,
        'queries': 85,       # Reduced baseline
        'accuracy': 85,      # Reduced baseline
        'verification': 80   # Reduced baseline
    }
    
    smart_dqi = sum(components[key] * smart_weights[key] for key in components)
    basic_dqi = sum(components[key] * BASE_WEIGHTS[key] for key in components)
    
    return smart_dqi, basic_dqi


# Process all studies
start_time = time.time()
all_results = []
successful_studies = 0
total_patients = 0

for i, folder in enumerate(all_study_folders, 1):
    context = context_patterns[i % len(context_patterns)]
    study_name = folder.replace('_', ' ')
    
    try:
        print(f"{i:2d}/{len(all_study_folders)} {study_name:30s} ... ", end='')
        study_start = time.time()
        
        # Aggregate study data
        df_agg = aggregate_study(folder, study_name, context)
        
        # Calculate DQI for all patients
        dqi_scores = []
        for _, row in df_agg.iterrows():
            smart_dqi, basic_dqi = calculate_dqi(row, context['therapeutic_area'], context['phase'])
            dqi_scores.append({
                'Smart_DQI': round(smart_dqi, 1),
                'Basic_DQI': round(basic_dqi, 1),
                'Difference': round(smart_dqi - basic_dqi, 1)
            })
        
        df_dqi = pd.DataFrame(dqi_scores)
        df_final = pd.concat([df_agg.reset_index(drop=True), df_dqi], axis=1)
        
        # Add risk levels with STRICT thresholds
        df_final['Smart_Risk'] = df_final['Smart_DQI'].apply(lambda x: 'Low' if x >= 92 else 'Medium' if x >= 80 else 'High')
        
        all_results.append(df_final)
        successful_studies += 1
        total_patients += len(df_final)
        
        study_time = time.time() - study_start
        print(f"{len(df_final):4d} pts | DQI {df_final['Smart_DQI'].mean():5.1f} | {study_time:.1f}s")
        
    except Exception as e:
        print(f"SKIP: {str(e)[:60]}")
        continue


# Check if we have any results
if len(all_results) == 0:
    print("\n❌ ERROR: No studies loaded successfully!")
    print("Check that your Excel files have 'Subject ID' and 'Site ID' columns.")
    exit(1)

# Combine all studies
final_df = pd.concat(all_results, ignore_index=True)
elapsed = time.time() - start_time

print("=" * 80)
print("✅ ALL STUDIES PROCESSED!")
print("=" * 80)
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
print(f"Studies processed: {successful_studies}/{len(all_study_folders)}")
print(f"Total patients: {total_patients:,}")
print(f"Speed: {total_patients/elapsed:.0f} patients/second")

print("\n📊 Risk Distribution:")
risk_counts = final_df['Smart_Risk'].value_counts()
for risk in ['High', 'Medium', 'Low']:
    if risk in risk_counts.index:
        count = risk_counts[risk]
        pct = count / len(final_df) * 100
        print(f"  {risk:8s}: {count:5d} ({pct:5.1f}%)")

print("\n📊 By Study:")
study_summary = final_df.groupby('Study').agg({
    'Subject_ID': 'count',
    'Smart_DQI': 'mean'
}).round(1)
study_summary.columns = ['Patients', 'Avg DQI']
print(study_summary.to_string())

# Save
final_df.to_csv('outputs/all_studies_smart_dqi.csv', index=False)
print(f"\n💾 Saved to: outputs/all_studies_smart_dqi.csv")
print("=" * 80)
