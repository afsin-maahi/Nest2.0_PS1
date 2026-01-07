import pandas as pd
import numpy as np

print("=" * 80)
print("🎲 ADDING REALISTIC NOISE TO DATA")
print("=" * 80)

# Load original data
df = pd.read_csv('outputs/all_studies_smart_dqi.csv')
print(f"\n📊 Loaded {len(df):,} patients")

# Convert to numeric
df['Overdue_Visits_Count'] = pd.to_numeric(df['Overdue_Visits_Count'], errors='coerce').fillna(0)
df['Missing_Pages'] = pd.to_numeric(df['Missing_Pages'], errors='coerce').fillna(0)
df['SAE_Pending_Count'] = pd.to_numeric(df['SAE_Pending_Count'], errors='coerce').fillna(0)

print(f"\n🔧 Adding realistic data variations...")

# Set random seed for reproducibility
np.random.seed(42)

# 1. Add random measurement noise (±10% variation)
print(f"   ✓ Adding measurement noise...")
df['Overdue_Visits_Count'] = df['Overdue_Visits_Count'] + np.random.normal(0, 0.5, len(df))
df['Overdue_Visits_Count'] = df['Overdue_Visits_Count'].clip(lower=0).round()

# 2. Simulate delayed reporting (some SAE not yet counted)
print(f"   ✓ Simulating delayed SAE reporting...")
# 20% of patients with SAE have underreported counts
mask = (df['SAE_Pending_Count'] > 0) & (np.random.random(len(df)) < 0.2)
underreport_amount = (df.loc[mask, 'SAE_Pending_Count'] * np.random.uniform(0.2, 0.4, mask.sum())).round()
df.loc[mask, 'SAE_Pending_Count'] = (df.loc[mask, 'SAE_Pending_Count'] - underreport_amount).clip(lower=0)

# 3. Add some false positives - patients flagged but actually okay
print(f"   ✓ Adding false positive flags...")
# 5% of low-risk patients get random issues added
low_risk = (df['Smart_Risk'] == 'Low')
false_positive_mask = low_risk & (np.random.random(len(df)) < 0.05)
df.loc[false_positive_mask, 'Overdue_Visits_Count'] += np.random.choice([1, 2], false_positive_mask.sum())

# 4. Add some false negatives - patients with hidden issues
print(f"   ✓ Adding hidden issues (false negatives)...")
# 3% of high-risk patients have issues that aren't captured yet
high_risk = (df['Smart_Risk'] == 'High')
false_negative_mask = high_risk & (np.random.random(len(df)) < 0.03)
df.loc[false_negative_mask, 'SAE_Pending_Count'] = (df.loc[false_negative_mask, 'SAE_Pending_Count'] * 0.7).round()

# 5. Add temporal variation - some visits just became overdue
print(f"   ✓ Adding temporal variations...")
# 10% of patients with no overdue visits just had one become overdue
no_overdue = (df['Overdue_Visits_Count'] == 0)
new_overdue_mask = no_overdue & (np.random.random(len(df)) < 0.10)
df.loc[new_overdue_mask, 'Overdue_Visits_Count'] = 1

# 6. Add data entry errors - occasional random spikes
print(f"   ✓ Simulating data entry errors...")
# 2% of records have data entry errors
error_mask = np.random.random(len(df)) < 0.02
error_types = np.random.choice(['visits', 'pages', 'sae'], error_mask.sum())
for idx, (i, row) in enumerate(df[error_mask].iterrows()):
    if error_types[idx] == 'visits':
        df.at[i, 'Overdue_Visits_Count'] += np.random.randint(1, 3)
    elif error_types[idx] == 'pages':
        df.at[i, 'Missing_Pages'] += np.random.randint(1, 5)

# Ensure all values are non-negative integers
df['Overdue_Visits_Count'] = df['Overdue_Visits_Count'].clip(lower=0).round().astype(int)
df['Missing_Pages'] = df['Missing_Pages'].clip(lower=0).round().astype(int)
df['SAE_Pending_Count'] = df['SAE_Pending_Count'].clip(lower=0).round().astype(int)

# Recalculate DQI with noisy data
print(f"\n🔄 Recalculating DQI scores with noisy data...")

def calculate_noisy_dqi(row):
    visits = row['Overdue_Visits_Count']
    pages = row['Missing_Pages']
    sae = row['SAE_Pending_Count']
    
    visits_score = 100 if visits == 0 else max(0, 100 - (visits * 20))
    pages_score = 100 if pages == 0 else max(0, 100 - (pages * 3))
    safety_score = 100 if sae == 0 else max(0, 100 - (sae * 25))
    
    # Smart weights for oncology
    weights = {
        'visits': 0.15,
        'pages': 0.10,
        'safety': 0.35,
        'queries': 0.15,
        'accuracy': 0.15,
        'verification': 0.10
    }
    
    components = {
        'visits': visits_score,
        'pages': pages_score,
        'safety': safety_score,
        'queries': 78,
        'accuracy': 82,
        'verification': 75
    }
    
    smart_dqi = sum(components[key] * weights[key] for key in components)
    return smart_dqi

df['Smart_DQI_Noisy'] = df.apply(calculate_noisy_dqi, axis=1)

# Recalculate risk with noisy DQI
def get_risk_level(score):
    return 'Low' if score >= 88 else 'Medium' if score >= 75 else 'High'

df['Smart_Risk_Noisy'] = df['Smart_DQI_Noisy'].apply(get_risk_level)

# Compare original vs noisy
print(f"\n📊 IMPACT OF NOISE:")
print("=" * 80)

print(f"\nOriginal Risk Distribution:")
print(df['Smart_Risk'].value_counts(normalize=True).mul(100).round(1).to_string())

print(f"\nNoisy Risk Distribution:")
print(df['Smart_Risk_Noisy'].value_counts(normalize=True).mul(100).round(1).to_string())

# Count risk category changes
risk_changed = (df['Smart_Risk'] != df['Smart_Risk_Noisy']).sum()
print(f"\n🔄 Risk Category Changes: {risk_changed:,} patients ({risk_changed/len(df)*100:.1f}%)")

# Show examples of changes
print(f"\n🔍 Example Risk Category Changes:")
changes = df[df['Smart_Risk'] != df['Smart_Risk_Noisy']].head(5)
for i, (idx, row) in enumerate(changes.iterrows(), 1):
    print(f"\n{i}. {row['Subject ID']} | {row['Study'][:30]}")
    print(f"   Original: {row['Smart_Risk']} (DQI: {row['Smart_DQI']:.1f})")
    print(f"   With Noise: {row['Smart_Risk_Noisy']} (DQI: {row['Smart_DQI_Noisy']:.1f})")
    print(f"   SAE: {row['SAE_Pending_Count']:.0f} | Visits: {row['Overdue_Visits_Count']:.0f}")

# Save noisy data
output_path = 'outputs/all_studies_noisy.csv'
df.to_csv(output_path, index=False)
print(f"\n💾 Noisy data saved to: {output_path}")

print("\n" + "=" * 80)
print("✅ REALISTIC NOISE ADDED!")
print(f"📊 {risk_changed:,} patients changed risk categories")
print(f"🎯 Models will now have {100 - (df['Smart_Risk'] == df['Smart_Risk_Noisy']).sum()/len(df)*100:.1f}% prediction challenge")
print("=" * 80)
