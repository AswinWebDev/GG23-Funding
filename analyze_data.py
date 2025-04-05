import pandas as pd
import numpy as np

# Load data
print("Loading data...")
historical = pd.read_csv('GG Allocation Since GG18.csv')
submission = pd.read_csv('submission.csv')

# Print column names to verify
print("\nHistorical data columns:")
print(historical.columns.tolist())

# Basic statistics of historical data
print("\nHistorical data contribution statistics:")
print(historical['Contribution Amount'].describe().round(2))

# Calculate community contributions in our predictions
print("\nPREDICTION ANALYSIS:")
results = {}
for round_type in ['WEB3 INFRA', 'DEV TOOLING', 'DAPPS & APPS', 'MATURE BUILDERS']:
    data = submission[submission['ROUND'] == round_type]
    count = len(data)
    total = data['AMOUNT'].sum()
    avg = data['AMOUNT'].mean()
    
    if round_type != 'MATURE BUILDERS':
        community_total = total - 200000
        community_avg = community_total / count
        results[round_type] = {
            'count': count, 
            'total': total, 
            'avg': avg, 
            'community_total': community_total, 
            'community_avg': community_avg
        }
    else:
        results[round_type] = {'count': count, 'total': total, 'avg': avg}

# Print results
for round_type, stats in results.items():
    print(f"\n{round_type}:")
    print(f"  Projects: {stats['count']}")
    print(f"  Total allocation: ${stats['total']:,.2f}")
    print(f"  Average per project: ${stats['avg']:,.2f}")
    if round_type != 'MATURE BUILDERS':
        print(f"  Community contributions: ${stats['community_total']:,.2f}")
        print(f"  Avg community contribution per project: ${stats['community_avg']:,.2f}")

# Simple categorization for historical rounds
print("\nHistorical round analysis:")
web3_keywords = ['Infrastructure', 'WEB3 INFRA']
dev_keywords = ['Developer', 'Tooling', 'DEV TOOLING', 'Libraries']
dapps_keywords = ['dApps', 'Apps', 'DAPPS']

def categorize_round(round_name):
    if any(keyword.lower() in round_name.lower() for keyword in web3_keywords):
        return 'WEB3 INFRA'
    elif any(keyword.lower() in round_name.lower() for keyword in dev_keywords):
        return 'DEV TOOLING'
    elif any(keyword.lower() in round_name.lower() for keyword in dapps_keywords):
        return 'DAPPS & APPS'
    return 'OTHER'

historical['Round_Category'] = historical['Round Name'].apply(categorize_round)

# Analyze by category
for category in ['WEB3 INFRA', 'DEV TOOLING', 'DAPPS & APPS']:
    cat_data = historical[historical['Round_Category'] == category]
    if len(cat_data) > 0:
        print(f"\n{category} historical analysis:")
        print(f"  Projects: {len(cat_data)}")
        print(f"  Total contribution amount: ${cat_data['Contribution Amount'].sum():,.2f}")
        print(f"  Average contribution per project: ${cat_data['Contribution Amount'].mean():,.2f}")
        print(f"  Median contribution per project: ${cat_data['Contribution Amount'].median():,.2f}")
        
# Compare our predictions with historical data
print("\nComparison of our predictions vs. historical data:")
for category in ['WEB3 INFRA', 'DEV TOOLING', 'DAPPS & APPS']:
    cat_data = historical[historical['Round_Category'] == category]
    if len(cat_data) > 0 and category in results:
        hist_avg = cat_data['Contribution Amount'].mean()
        our_avg = results[category]['community_avg']
        diff_pct = ((our_avg - hist_avg) / hist_avg) * 100 if hist_avg > 0 else float('inf')
        
        print(f"\n{category}:")
        print(f"  Historical avg contribution: ${hist_avg:,.2f}")
        print(f"  Our predicted avg contribution: ${our_avg:,.2f}")
        print(f"  Difference: {diff_pct:,.1f}%")
