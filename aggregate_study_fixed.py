import pandas as pd
import os

def aggregate_study(study_folder, study_name):
    """
    Aggregate all data for a single study
    Returns DataFrame with Subject_ID, Site_ID, and all metrics
    """
    print("=" * 80)
    print(f"📊 AGGREGATING {study_name}")
    print("=" * 80)
    
    base_path = 'data/raw_studies'
    study_path = f'{base_path}/{study_folder}'
    files = os.listdir(study_path)
    
    # 1. Load main EDC metrics - SKIP first row, use row 2 as header
    edc_files = [f for f in files if 'CPID' in f or 'EDC' in f]
    if not edc_files:
        raise Exception("No EDC file found")
    
    print(f"✓ Loading {edc_files[0][:50]}...")
    
    # Load with header=0 to capture multi-level headers
    df_main = pd.read_excel(f'{study_path}/{edc_files[0]}', sheet_name=1, header=0)
    
    # The first row contains metric labels like "# Missing Pages"
    # We need to check if row 1 is labels and row 2+ is data
    first_row = df_main.iloc[0]
    
    # If first row contains strings like "# Missing Pages", it's a label row
    if any('# Missing' in str(val) or '# Site' in str(val) for val in first_row.values if pd.notna(val)):
        print(f"  → Detected multi-level headers, using row 2 as data start")
        # Use the first row values as the actual column names
        new_cols = []
        for i, col in enumerate(df_main.columns):
            label = first_row.iloc[i]
            if pd.notna(label) and (isinstance(label, str)) and ('#' in label or 'Missing' in label or 'Queries' in label):
                new_cols.append(label.strip())
            else:
                new_cols.append(col)
        
        df_main.columns = new_cols
        df_main = df_main.iloc[1:].reset_index(drop=True)  # Skip the label row
    
    df_main.columns = df_main.columns.str.strip()
    
    print(f"  → Raw shape: {df_main.shape}")
    print(f"  → Columns: {df_main.columns.tolist()[:5]}")
    
    # Find columns
    subject_col = None
    site_col = None
    missing_pages_col = None
    site_queries_col = None
    monitor_queries_col = None
    non_conformant_col = None
    
    for col in df_main.columns:
        col_str = str(col).lower()
        if 'subject' in col_str and 'id' in col_str:
            subject_col = col
        elif 'site' in col_str and 'id' in col_str:
            site_col = col
        elif '# missing pages' in col_str or 'missing page' in col_str:
            missing_pages_col = col
        elif '# site queries' in col_str or 'site queries' in col_str:
            site_queries_col = col
        elif '# field monitor queries' in col_str or 'monitor queries' in col_str:
            monitor_queries_col = col
        elif '# pages with non-conformant' in col_str or 'non-conformant' in col_str:
            non_conformant_col = col
    
    if not subject_col or not site_col:
        raise Exception(f"Missing Subject/Site ID columns. Available: {df_main.columns.tolist()}")
    
    print(f"  → Found Subject: '{subject_col}'")
    print(f"  → Found Site: '{site_col}'")
    if missing_pages_col:
        print(f"  → Found Missing Pages: '{missing_pages_col}'")
    if site_queries_col:
        print(f"  → Found Site Queries: '{site_queries_col}'")
    if non_conformant_col:
        print(f"  → Found Non-Conformant: '{non_conformant_col}'")
    
    # Rename columns
    rename_dict = {
        subject_col: 'Subject_ID',
        site_col: 'Site_ID'
    }
    if missing_pages_col:
        rename_dict[missing_pages_col] = 'Missing_Pages'
    if site_queries_col:
        rename_dict[site_queries_col] = 'Site_Queries'
    if monitor_queries_col:
        rename_dict[monitor_queries_col] = 'Monitor_Queries'
    if non_conformant_col:
        rename_dict[non_conformant_col] = 'Non_Conformant_Pages'
    
    df_clean = df_main.rename(columns=rename_dict)
    
    # Clean Subject_ID
    df_clean = df_clean[df_clean['Subject_ID'].notna()].copy()
    df_clean['Subject_ID'] = df_clean['Subject_ID'].astype(str)
    
    # Remove any rows that are still headers
    df_clean = df_clean[~df_clean['Subject_ID'].str.contains('Subject ID', case=False, na=False)]
    
    print(f"  → Sample Subject IDs: {df_clean['Subject_ID'].head(5).tolist()}")
    print(f"  → After cleaning: {len(df_clean)} patients")
    
    # Initialize/clean metrics
    if 'Overdue_Visits_Count' not in df_clean.columns:
        df_clean['Overdue_Visits_Count'] = 0
        
    if 'Missing_Pages' not in df_clean.columns:
        df_clean['Missing_Pages'] = 0
        print(f"  ⚠️  Missing Pages column not found - defaulting to 0")
    else:
        df_clean['Missing_Pages'] = pd.to_numeric(df_clean['Missing_Pages'], errors='coerce').fillna(0)
        total_missing = df_clean['Missing_Pages'].sum()
        avg_missing = df_clean['Missing_Pages'].mean()
        if total_missing > 0:
            print(f"  ✓ Found {total_missing:.0f} total missing pages (avg: {avg_missing:.1f} per patient)")
        else:
            print(f"  ✓ No missing pages - all pages complete!")
    
    # Add queries data if available
    if 'Site_Queries' in df_clean.columns:
        df_clean['Site_Queries'] = pd.to_numeric(df_clean['Site_Queries'], errors='coerce').fillna(0)
        total_queries = df_clean['Site_Queries'].sum()
        if total_queries > 0:
            print(f"  ✓ Found {total_queries:.0f} total site queries")
    
    if 'Non_Conformant_Pages' in df_clean.columns:
        df_clean['Non_Conformant_Pages'] = pd.to_numeric(df_clean['Non_Conformant_Pages'], errors='coerce').fillna(0)
        total_nc = df_clean['Non_Conformant_Pages'].sum()
        if total_nc > 0:
            print(f"  ✓ Found {total_nc:.0f} non-conformant pages")
    
    df_clean['SAE_Pending_Count'] = 0
    
    # 2. Load Visit Tracker - Count OVERDUE visits only
    visit_files = [f for f in files if 'Visit' in f and ('Projection' in f or 'Tracker' in f)]
    if visit_files:
        try:
            df_visits = pd.read_excel(f'{study_path}/{visit_files[0]}', sheet_name=0, header=0)
            if len(df_visits) > 0:
                # Find subject column
                visit_subj_col = None
                for col in df_visits.columns:
                    if 'Subject' in col:
                        visit_subj_col = col
                        break
                
                # Find days outstanding column
                days_outstanding_col = None
                for col in df_visits.columns:
                    if 'Days Outstanding' in col or 'Outstanding' in col:
                        days_outstanding_col = col
                        break
                
                if visit_subj_col and days_outstanding_col:
                    # Count overdue visits (Days Outstanding > 0)
                    df_visits[visit_subj_col] = df_visits[visit_subj_col].astype(str)
                    df_visits[days_outstanding_col] = pd.to_numeric(df_visits[days_outstanding_col], errors='coerce')
                    
                    # Filter to overdue visits only (days outstanding > 0)
                    df_overdue = df_visits[df_visits[days_outstanding_col] > 0].copy()
                    
                    if len(df_overdue) > 0:
                        visits_count = df_overdue.groupby(visit_subj_col).size().reset_index(name='Overdue_Visits_Count')
                        visits_count.columns = ['Subject_ID', 'Overdue_Visits_Count']
                        
                        df_clean = df_clean.merge(visits_count, on='Subject_ID', how='left', suffixes=('', '_new'))
                        if 'Overdue_Visits_Count_new' in df_clean.columns:
                            df_clean['Overdue_Visits_Count'] = df_clean['Overdue_Visits_Count_new'].fillna(0)
                            df_clean = df_clean.drop(columns=['Overdue_Visits_Count_new'])
                        
                        total_overdue = len(df_overdue)
                        avg_days = df_overdue[days_outstanding_col].mean()
                        print(f"✓ Found {total_overdue} overdue visits (avg: {avg_days:.0f} days late)")
                    else:
                        print(f"✓ No overdue visits - all visits on schedule!")
                elif visit_subj_col:
                    # Fallback: Just count all visits if no days outstanding column
                    df_visits[visit_subj_col] = df_visits[visit_subj_col].astype(str)
                    visits_count = df_visits.groupby(visit_subj_col).size().reset_index(name='Overdue_Visits_Count')
                    visits_count.columns = ['Subject_ID', 'Overdue_Visits_Count']
                    df_clean = df_clean.merge(visits_count, on='Subject_ID', how='left', suffixes=('', '_new'))
                    if 'Overdue_Visits_Count_new' in df_clean.columns:
                        df_clean['Overdue_Visits_Count'] = df_clean['Overdue_Visits_Count_new'].fillna(0)
                        df_clean = df_clean.drop(columns=['Overdue_Visits_Count_new'])
                    print(f"⚠️  Counting all {len(df_visits)} visit records (no 'Days Outstanding' column)")
        except Exception as e:
            print(f"⚠️  Visit processing: {str(e)[:50]}")
    
    # 3. Load SAE Dashboard
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
                    print(f"✓ Found {len(df_sae_pending)} pending SAE reviews")
        except Exception as e:
            pass
    
    # Final cleanup
    df_clean['Overdue_Visits_Count'] = pd.to_numeric(df_clean['Overdue_Visits_Count'], errors='coerce').fillna(0)
    df_clean['Missing_Pages'] = pd.to_numeric(df_clean['Missing_Pages'], errors='coerce').fillna(0)
    df_clean['SAE_Pending_Count'] = pd.to_numeric(df_clean['SAE_Pending_Count'], errors='coerce').fillna(0)
    
    # Add study info
    df_clean['Study'] = study_name
    
    print(f"\n✅ {study_name} aggregation complete!")
    print(f"   Patients: {len(df_clean)}")
    print(f"   Avg Overdue Visits: {df_clean['Overdue_Visits_Count'].mean():.1f}")
    print(f"   Avg Missing Pages: {df_clean['Missing_Pages'].mean():.1f}")
    print(f"   Avg Pending SAE: {df_clean['SAE_Pending_Count'].mean():.1f}")
    
    # Rename back for compatibility
    df_clean = df_clean.rename(columns={'Subject_ID': 'Subject ID', 'Site_ID': 'Site ID'})
    
    final_cols = ['Subject ID', 'Site ID', 'Study', 'Overdue_Visits_Count', 'Missing_Pages', 'SAE_Pending_Count']
    
    return df_clean[final_cols].copy()
