import pandas as pd
import os

# Check available columns in all studies
base_path = 'data/raw_studies'
all_folders = sorted(os.listdir(base_path))

print("=" * 80)
print("📋 CHECKING ALL EXCEL COLUMNS")
print("=" * 80)

# Check first 3 studies in detail
for study_folder in all_folders[:3]:
    study_path = f'{base_path}/{study_folder}'
    files = os.listdir(study_path)
    edc_files = [f for f in files if 'CPID' in f or 'EDC' in f]
    
    if not edc_files:
        continue
    
    print(f"\n{'=' * 80}")
    print(f"📁 STUDY: {study_folder}")
    print(f"{'=' * 80}")
    print(f"File: {edc_files[0]}\n")
    
    file_path = f'{study_path}/{edc_files[0]}'
    
    # Load the EDC metrics sheet
    df = pd.read_excel(file_path, sheet_name=1, header=0)
    
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(f"ALL COLUMNS:")
    print("-" * 80)
    
    for i, col in enumerate(df.columns, 1):
        # Count non-null values
        non_null = df[col].notna().sum()
        sample = df[col].dropna().iloc[0] if non_null > 0 else "No data"
        print(f"{i:2d}. {col:40s} | Non-null: {non_null:4d} | Sample: {sample}")
    
    # Check for specific data quality columns
    print(f"\n🔎 DATA QUALITY RELATED COLUMNS:")
    print("-" * 80)
    keywords = ['missing', 'page', 'incomplete', 'pending', 'overdue', 'query', 
                'queries', 'non-conform', 'error', 'issue', 'open', 'closed']
    
    found = False
    for col in df.columns:
        if any(keyword in str(col).lower() for keyword in keywords):
            found = True
            print(f"✓ {col}")
            # Show distribution of values
            value_counts = df[col].value_counts().head(3)
            if len(value_counts) > 0:
                print(f"  Top values: {value_counts.to_dict()}")
    
    if not found:
        print("❌ No data quality metric columns found!")

print("\n" + "=" * 80)
print("✅ CHECK COMPLETE - SUMMARY")
print("=" * 80)
print("\nBased on the output above, we can confirm:")
print("1. Whether 'Missing Pages' column exists")
print("2. What other data quality metrics are available")
print("3. What the actual column names are")
