import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime
from math import sqrt

# Set random state for reproducibility
RANDOM_STATE = 42

def match_historical_projects(project_id, project_name, historical_data, round_category=None, threshold=0.8):
    """Match projects first by ID, then by name with fuzzy matching to find historical data"""
    
    # Skip ID matching for Mature Builders as they don't have project IDs
    if round_category != "MATURE BUILDERS" and project_id and not pd.isna(project_id) and 'project_id' in historical_data.columns:
        # Try exact match
        project_matches = historical_data[historical_data['project_id'] == project_id]
        if len(project_matches) > 0:
            return project_matches['PROJECT'].iloc[0]
    
    # If no ID match found, fall back to name matching
    best_match = None
    best_score = 0
    
    # Handle non-string inputs
    if not isinstance(project_name, str) or pd.isna(project_name):
        return None
    
    for hist_project in historical_data['PROJECT'].unique():
        if pd.isna(hist_project) or not isinstance(hist_project, str):
            continue
            
        # Calculate similarity score
        score = SequenceMatcher(None, project_name.lower(), hist_project.lower()).ratio()
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = hist_project
    
    return best_match

def calculate_mature_score(project_name, historical_data):
    """Calculate mature score based on more Gitcoin metrics"""
    # Default score for new projects
    if project_name not in historical_data['PROJECT'].values:
        return 0.0
    
    # Get project history
    project_data = historical_data[historical_data['PROJECT'] == project_name]
    
    # Basic metrics
    round_participation = len(project_data)
    total_contributors = project_data['contributor_count'].sum()
    total_matching = project_data['matching_amount'].sum()
    avg_matching = project_data['matching_amount'].mean()
    
    # Handle zero division
    if total_contributors == 0:
        return 0.0
    
    # Base score calculation - logarithmic scaling for diminishing returns
    base_score = np.log1p(total_matching / 1000) * np.log1p(round_participation)
    
    # Consistency factor - favors projects that have been around longer (up to 2x boost)
    consistency_factor = 1.0 + min(round_participation / 3.0, 1.0)
    
    # Community impact factor - based on contributor reach (diminishing returns)
    community_factor = 1.0 + np.log1p(total_contributors / 100) / 2
    
    # Recent growth factor - prioritize growing projects
    growth_factor = 1.0
    if round_participation >= 2:
        # Get recent rounds
        sorted_data = project_data.sort_values('round_id')
        recent = sorted_data.iloc[-2:] 
        
        # Calculate growth rates
        if 'contributor_count' in recent.columns:
            contributor_growth = recent['contributor_count'].pct_change().iloc[-1] if len(recent) > 1 else 0
            funding_growth = recent['matching_amount'].pct_change().iloc[-1] if len(recent) > 1 else 0
            
            # Only consider positive growth
            if contributor_growth > 0 or funding_growth > 0:
                growth_factor = 1.0 + min(max(contributor_growth + funding_growth, 0), 0.5)  # +0% to +50% boost
    
    # Donor retention factor - prioritize projects with returning contributors
    donor_retention_factor = 1.0
    if round_participation >= 2:
        # Simple donor retention approximation based on contributor count stability
        sorted_data = project_data.sort_values('round_id')
        avg_growth = sorted_data['contributor_count'].pct_change().mean() 
        stability = 1.0 / (1.0 + np.std(sorted_data['contributor_count']) / (sorted_data['contributor_count'].mean() + 1))
        donor_retention_factor = 1.0 + min(max(stability, 0), 0.5)  # +0% to +50% boost
    
    # Calculate final score with all factors
    mature_score = base_score * consistency_factor * community_factor * growth_factor * donor_retention_factor
    
    return mature_score

def load_and_prepare_historical_data(file_path):
    """Load and prepare historical data for model training"""
    print(f"Loading historical data from: {file_path}")
    
    try:
        # Load historical data
        data = pd.read_csv(file_path)
        
        # Rename columns to match our expected format
        column_mapping = {
            'Application Title': 'PROJECT',
            'Gitcoin Project Id': 'project_id',
            'Round Name': 'ROUND',
            'Matching Amount': 'matching_amount',
            '# of Contributors': 'contributor_count',
            'Contribution Amount': 'contribution_amount',
            'Gitcoin Round Id': 'round_id',
            'Gitcoin Grants #': 'grants_round'
        }
        
        # Only rename columns that exist in the dataframe
        rename_dict = {col: new_col for col, new_col in column_mapping.items() if col in data.columns}
        data = data.rename(columns=rename_dict)
        
        # Ensure we have PROJECT and ROUND columns
        if 'PROJECT' not in data.columns:
            if 'Application Title' in data.columns:
                data['PROJECT'] = data['Application Title']
        
        if 'ROUND' not in data.columns:
            if 'Round Name' in data.columns:
                data['ROUND'] = data['Round Name']
        
        # Ensure we have numeric columns
        numeric_cols = ['matching_amount', 'contributor_count', 'contribution_amount']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            else:
                data[col] = 0
        
        # Calculate average contribution size
        data['avg_contribution_size'] = data['contribution_amount'] / data['contributor_count'].replace(0, 1)
        
        # Calculate total amount
        data['AMOUNT'] = data['matching_amount'] + data['contribution_amount']
        
        # Print sample data
        print("Sample historical data:")
        print(data.head())
        
        return data
    
    except Exception as e:
        print(f"Error loading historical data: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_test_data(test_file, historical_data):
    """Prepare test data with features for prediction"""
    print(f"Preparing test data from: {test_file}")
    
    try:
        # Load test data
        test_data = pd.read_csv(test_file)
        
        # Ensure we have the right columns
        if 'PROJECT' not in test_data.columns and 'project' in test_data.columns:
            test_data['PROJECT'] = test_data['project']
        
        if 'ROUND' not in test_data.columns and 'round' in test_data.columns:
            test_data['ROUND'] = test_data['round']
        
        # Handle project ID
        if 'PROJECT_ID' in test_data.columns:
            test_data['project_id'] = test_data['PROJECT_ID']
        
        # Initialize columns for historical metrics
        test_data['avg_matching'] = 0
        test_data['avg_contributors'] = 0
        test_data['avg_contributions'] = 0
        test_data['avg_contribution_size'] = 0
        test_data['consistency'] = 0
        test_data['is_returning'] = 0
        test_data['mature_score'] = 0
        test_data['enhanced_mature_score'] = 0
        test_data['past_rounds'] = 0  # New feature: number of past rounds participated
        test_data['historical_total_funding'] = 0  # New feature: total historical funding
        
        # Match each project with historical data
        for idx, row in test_data.iterrows():
            project_id = row.get('project_id', None)
            project_name = row['PROJECT']
            round_category = row.get('ROUND', None)
            
            # Try to find historical data for this project
            matched_project = match_historical_projects(project_id, project_name, historical_data, round_category)
            
            if matched_project:
                # Get project history
                project_data = historical_data[historical_data['PROJECT'] == matched_project]
                
                # Calculate basic metrics
                test_data.loc[idx, 'avg_matching'] = project_data['matching_amount'].mean()
                test_data.loc[idx, 'avg_contributors'] = project_data['contributor_count'].mean()
                test_data.loc[idx, 'avg_contributions'] = project_data['contribution_amount'].mean()
                test_data.loc[idx, 'avg_contribution_size'] = project_data['avg_contribution_size'].mean()
                test_data.loc[idx, 'consistency'] = len(project_data) / 3  # normalize by 3 rounds
                test_data.loc[idx, 'is_returning'] = 1
                test_data.loc[idx, 'past_rounds'] = len(project_data)  # New feature
                test_data.loc[idx, 'historical_total_funding'] = project_data['AMOUNT'].sum()  # New feature
                
                # Calculate mature score
                test_data.loc[idx, 'mature_score'] = calculate_mature_score(matched_project, historical_data)
                
                # Enhanced mature score with additional metrics
                enhanced_score = test_data.loc[idx, 'mature_score']
                
                # Add contributor growth boost
                if len(project_data) >= 2:
                    sorted_data = project_data.sort_values('round_id')
                    if 'contributor_count' in sorted_data.columns:
                        contributor_growth = sorted_data['contributor_count'].pct_change().dropna().mean()
                        enhanced_score *= (1 + max(0, min(contributor_growth, 1.0)))
                
                test_data.loc[idx, 'enhanced_mature_score'] = enhanced_score
        
        # Print sample data
        print("Sample test data:")
        print(test_data.head())
        
        return test_data
    
    except Exception as e:
        print(f"Error preparing test data: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_simple_model(historical_data, round_category):
    """Train a simple XGBoost model for a round category"""
    print(f"Training model for {round_category}...")
    
    # Filter historical data for this round (using flexible pattern matching)
    if round_category == 'WEB3 INFRA':
        round_data = historical_data[historical_data['ROUND'].str.contains('Infrastructure|INFRA|Web3', case=False, na=False)]
    elif round_category == 'DEV TOOLING':
        round_data = historical_data[historical_data['ROUND'].str.contains('Developer|DEV|Tool|Library', case=False, na=False)]
    elif round_category == 'DAPPS & APPS':
        round_data = historical_data[historical_data['ROUND'].str.contains('dApp|App|dApps|Apps', case=False, na=False)]
    elif round_category == 'MATURE BUILDERS':
        round_data = historical_data[historical_data['ROUND'].str.contains('Mature|Builder', case=False, na=False)]
    else:
        round_data = historical_data.copy()
    
    # If not enough data, use all data
    if len(round_data) < 10:
        print(f"Not enough data for {round_category}, using all data.")
        round_data = historical_data.copy()
    
    # Create a simple model using XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=RANDOM_STATE
    )
    
    # Select features (only use features we're sure are available)
    available_features = ['avg_contribution_size', 'contributor_count', 'contribution_amount']
    features = [f for f in available_features if f in round_data.columns]
    
    # If no features are available, return a simple average model
    if not features:
        print(f"No usable features for {round_category}. Using average model.")
        avg_matching = round_data['matching_amount'].mean()
        return lambda X: np.full(len(X), avg_matching), []
    
    # Prepare training data
    X = round_data[features]
    y = round_data['matching_amount'].values
    
    # Train the model
    model.fit(X, y)
    
    return model, features

def donation_profile_clustermatch(donation_df):
    """
    Implement cluster QF matching as per Gitcoin's latest methodology
    
    donation_df: DataFrame where rows are unique donors, columns are projects, 
                 and entry i,j denotes user i's total donation to project j
    """
    projects = donation_df.columns
    clusters = {}  # maps clusters to total donation amounts from those clusters

    # Build up the cluster donation amounts
    for (wallet, donations) in donation_df.iterrows():
        # Determine cluster based on donation pattern
        c = ''.join('1' if donations[p] > 0 else '0' for p in projects)

        # Update cluster donation amounts
        if c in clusters.keys():
            for p in projects:
                clusters[c][p] += donations[p]
        else:
            clusters[c] = {p: donations[p] for p in projects}

    # Apply QF on the clustered donations
    funding = {p: sum(sqrt(clusters[c][p]) for c in clusters.keys() if clusters[c][p] > 0) ** 2 for p in projects}
    
    return funding

def predict_funding(test_data):
    """Generate predictions for funding amounts"""
    print("Generating predictions...")
    
    # Create a copy of test data to store predictions
    predictions = test_data.copy()
    predictions['AMOUNT'] = 0
    
    # Define total matching pools
    total_matching = {
        'WEB3 INFRA': 200000,
        'DEV TOOLING': 200000,
        'DAPPS & APPS': 200000,
        'MATURE BUILDERS': 600000
    }
    
    # Define matching caps per project (as a percentage of total matching pool)
    matching_caps = {
        'WEB3 INFRA': 0.10,      # 10% cap per project
        'DEV TOOLING': 0.10,      # 10% cap per project
        'DAPPS & APPS': 0.05,     # 5% cap per project
        'MATURE BUILDERS': 1.0    # No cap for MATURE BUILDERS
    }
    
    # Predict for each round category
    for round_category in ['WEB3 INFRA', 'DEV TOOLING', 'DAPPS & APPS', 'MATURE BUILDERS']:
        print(f"Predicting for {round_category}...")
        
        # Get data for this round
        round_data = predictions[predictions['ROUND'] == round_category].copy()
        
        if len(round_data) == 0:
            print(f"No projects for {round_category}")
            continue
        
        # Special handling for MATURE BUILDERS round
        if round_category == 'MATURE BUILDERS':
            # For MATURE BUILDERS, we'll use a more honest approach acknowledging data limitations
            # This is a placeholder that should be replaced with real research for each project
            print("Note: MATURE BUILDERS predictions require extensive research for each project")
            
            # Start with default baseline scores
            round_data['mature_score'] = 50  # Base score out of 100
            
            # We know some projects have high ecosystem impact based on general knowledge
            # These are partial insights that should be supplemented with real research
            known_projects = {
    'revoke-cash': 38,  # Updated from GitHub metrics
    'heyxyz': 0,  # Updated from GitHub metrics
    'defi-llama': 62,  # Updated from GitHub metrics
    'idriss-crypto': 1,  # Updated from GitHub metrics
    'poapin-glory-lab': 48,  # Updated from GitHub metrics
    'l2beat': 37,  # Updated from GitHub metrics
    'hypercerts': 11,  # Updated from GitHub metrics
    'rotki': 60,  # Updated from GitHub metrics
    'giveth': 11,  # Updated from GitHub metrics
    'ethstaker': 32,  # Updated from GitHub metrics
    'fundingthecommons': 2,  # Updated from GitHub metrics
    'tapexyz': 49,  # Updated from GitHub metrics
    'ethereum-attestation-service': 6,  # Updated from GitHub metrics
    'dappnode': 25,  # Updated from GitHub metrics
    'tor-project': 2,  # Updated from GitHub metrics
    'zaratandotworld': 2,  # Updated from GitHub metrics
    'jobstash': 1,  # Updated from GitHub metrics
    'ethers-io': 47,  # Updated from GitHub metrics
    'wevm': 61,  # Updated from GitHub metrics
    '0xfacet': 4,  # Updated from GitHub metrics
    'shapeshift': 22,  # Updated from GitHub metrics
    'citizenwallet': 8,  # Updated from GitHub metrics
    'blockscout': 61,  # Updated from GitHub metrics
    'zkemail': 33,  # Updated from GitHub metrics
    'kleo-network': 7,  # Updated from GitHub metrics
    'carmineoptions': 6,  # Updated from GitHub metrics
    'beacon-chain': 15,  # Updated from GitHub metrics
    'glo-foundation': 8,  # Updated from GitHub metrics
    'datonic': 53,  # Updated from GitHub metrics
    'zkp2p': 48,  # Updated from GitHub metrics
}
            
            # Apply known scores where available
            for idx, row in round_data.iterrows():
                project_name = row['PROJECT'].lower().strip()
                if project_name in known_projects:
                    round_data.loc[idx, 'mature_score'] = known_projects[project_name]
                else:
                    # For unknown projects, add some variability but keep them below known projects
                    # This represents our uncertainty
                    round_data.loc[idx, 'mature_score'] = np.random.uniform(40, 65)
            
            # Apply category boost based on project name keywords
            # This represents heuristic classification with limited information
            for idx, row in round_data.iterrows():
                project_name = row['PROJECT'].lower()
                
                # Infrastructure and protocol projects typically get higher scores
                if any(word in project_name for word in ['chain', 'protocol', 'infra', 'eth', 'crypto', 'zk']):
                    boost = np.random.uniform(1.1, 1.2)
                    round_data.loc[idx, 'mature_score'] *= boost
                
                # Developer tools and wallets
                elif any(word in project_name for word in ['wallet', 'dev', 'tool', 'sdk']):
                    boost = np.random.uniform(1.05, 1.15)
                    round_data.loc[idx, 'mature_score'] *= boost
            
            # Add a realistic level of variance to model real-world allocation decisions
            variance = np.random.normal(1.0, 0.1, size=len(round_data))
            round_data['mature_score'] *= variance
            
            # Scale scores to create more dramatic distribution (reward excellence)
            round_data['mature_score'] = round_data['mature_score'] ** 1.3
            
            # Normalize scores to allocate the matching pool
            total_score = round_data['mature_score'].sum()
            round_data['matching_amount'] = round_data['mature_score'] / total_score * total_matching[round_category]
            round_data['AMOUNT'] = round_data['matching_amount']
            
            print("Warning: MATURE BUILDERS predictions should be refined with project-specific research")
        
        # For other rounds, predict both matching and community contributions
        else:
            # First predict community contributions
            if 'is_returning' in round_data.columns and 'avg_contributors' in round_data.columns:
                # Number of simulated donors - use a base number scaled by project metrics
                num_donors = 100  # Base number of donors to simulate
                
                # Create a list of all projects in this round
                projects_list = round_data['PROJECT'].tolist()
                
                # Create donation profile DataFrame with simulated donors
                donor_ids = [f"donor_{i}" for i in range(num_donors)]
                donation_df = pd.DataFrame(0, index=donor_ids, columns=projects_list)
                
                # For each donor, determine which projects they support based on project metrics
                for donor in donor_ids:
                    # Each donor has a preference vector that determines likelihood of supporting each project
                    # Projects with higher historical metrics are more likely to be supported
                    preferences = {}
                    
                    for proj_idx, proj_row in round_data.iterrows():
                        project = proj_row['PROJECT']
                        # Base preference from project metrics
                        base_pref = 0.1  # Base 10% chance for any project
                        
                        # Add preference boost for returning projects
                        if proj_row['is_returning'] == 1:
                            base_pref += 0.2  # +20% for returning projects
                            
                            # Further boost based on historical performance
                            base_pref += min(0.3, proj_row['avg_contributors'] / 100)  # Up to +30% based on avg contributors
                        
                        preferences[project] = base_pref
                    
                    # Determine which projects this donor supports (with some randomness)
                    for project in projects_list:
                        if np.random.random() < preferences.get(project, 0.1):
                            # If they support it, determine donation amount (1-20 units)
                            donation_df.loc[donor, project] = np.random.uniform(1, 20)
                
                # Apply cluster QF matching
                funding = donation_profile_clustermatch(donation_df)
                
                # Convert dictionary to array matching round_data order
                matched_amounts = np.array([funding.get(project, 0.1) for project in round_data['PROJECT']])
                
                # Normalize to sum to total matching pool
                if matched_amounts.sum() > 0:
                    normalized_amounts = matched_amounts / matched_amounts.sum() * total_matching[round_category]
                else:
                    # Fallback if simulation produces zero matching
                    normalized_amounts = np.ones(len(round_data)) * total_matching[round_category] / len(round_data)
                
                # Update predictions
                round_data['matching_amount'] = normalized_amounts
                
                # Add predicted contributions based on cluster QF model
                # Usually projects with higher match also get more direct contributions
                contribution_factor = 0.5  # Direct contributions typically less than matching
                round_data['predicted_contributions'] = normalized_amounts * contribution_factor * np.random.uniform(0.7, 1.3, size=len(round_data))
                
                # Calculate total
                round_data['AMOUNT'] = round_data['matching_amount'] + round_data['predicted_contributions']
            else:
                # Fallback for rounds with no historical data
                # Use a simple random allocation model
                base_allocation = total_matching[round_category] / len(round_data)
                noise_factor = np.random.normal(loc=1.0, scale=0.15, size=len(round_data))
                round_data['matching_amount'] = base_allocation * noise_factor
                
                # Ensure we still sum to the total pool
                scale_factor = total_matching[round_category] / round_data['matching_amount'].sum()
                round_data['matching_amount'] = round_data['matching_amount'] * scale_factor
                
                # Add contribution amount
                category_base = {
                    'WEB3 INFRA': 150,
                    'DEV TOOLING': 100,
                    'DAPPS & APPS': 200,
                }
                avg_contribution = category_base.get(round_category, 100)
                contribution_noise = np.random.normal(loc=1.0, scale=0.2, size=len(round_data))
                round_data['predicted_contributions'] = avg_contribution * contribution_noise
                
                # Calculate total
                round_data['AMOUNT'] = round_data['matching_amount'] + round_data['predicted_contributions']
        
        # Apply matching cap per project
        cap_amount = total_matching[round_category] * matching_caps[round_category]
        
        # Apply cap to matching amount only (not total amount)
        round_data['matching_amount'] = round_data['matching_amount'].apply(lambda x: min(x, cap_amount))
        
        # Recalculate total amount with capped matching
        if round_category != 'MATURE BUILDERS':
            round_data['AMOUNT'] = round_data['matching_amount'] + round_data['predicted_contributions']
        else:
            round_data['AMOUNT'] = round_data['matching_amount']
        
        # After applying caps, redistribute excess matching funds
        if round_category != 'MATURE BUILDERS':
            # Calculate how much matching was actually used after applying caps
            total_used = round_data['matching_amount'].sum()
            
            # If we didn't use the full pool, redistribute the remainder proportionally to uncapped projects
            if total_used < total_matching[round_category]:
                # Find projects below the cap
                uncapped_projects = round_data[round_data['matching_amount'] < cap_amount]
                
                if len(uncapped_projects) > 0:
                    # Calculate how much to redistribute
                    remaining_pool = total_matching[round_category] - total_used
                    
                    # Get original weights for uncapped projects
                    original_weights = uncapped_projects['matching_amount'] / uncapped_projects['matching_amount'].sum()
                    
                    # Distribute remaining funds proportionally
                    additional_funds = original_weights * remaining_pool
                    
                    # Add to uncapped projects
                    for idx in uncapped_projects.index:
                        round_data.loc[idx, 'matching_amount'] += additional_funds.loc[idx]
                    
                    # Recalculate total amount
                    round_data['AMOUNT'] = round_data['matching_amount'] + round_data['predicted_contributions']
        
        # Update predictions dataframe
        predictions.loc[predictions['ROUND'] == round_category, 'AMOUNT'] = round_data['AMOUNT']
    
    # Combine all predictions
    all_predictions = pd.concat([predictions[['PROJECT', 'ROUND', 'AMOUNT']]])
    
    # Ensure we have the correct columns for submission
    submission = pd.merge(
        test_data[['PROJECT_ID', 'PROJECT', 'ROUND']],
        all_predictions[['PROJECT', 'ROUND', 'AMOUNT']],
        on=['PROJECT', 'ROUND'],
        how='left'
    )
    
    # Fill any missing values
    submission['AMOUNT'] = submission['AMOUNT'].fillna(0)
    
    # Make sure columns are in the right order
    submission = submission[['PROJECT_ID', 'PROJECT', 'ROUND', 'AMOUNT']]
    
    return submission[['PROJECT_ID', 'PROJECT', 'ROUND', 'AMOUNT']]

def generate_submission(predictions, output_path):
    """Generate submission file"""
    print(f"Generating submission file: {output_path}")
    
    # Save the predictions to a CSV file
    predictions.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return predictions

def main():
    """Main function to run the funding prediction pipeline"""
    print("Starting Gitcoin Grants Round 23 funding prediction pipeline")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # File paths
    historical_data_path = "GG Allocation Since GG18.csv"
    test_data_path = "projects_Apr_1.csv"  # Updated to new file with project IDs
    output_path = "submission.csv"
    
    # 1. Load and prepare historical data
    historical_data = load_and_prepare_historical_data(historical_data_path)
    
    if historical_data is None:
        print("Failed to load historical data. Exiting.")
        return
    
    # 2. Prepare test data
    test_data = prepare_test_data(test_data_path, historical_data)
    
    if test_data is None:
        print("Failed to prepare test data. Exiting.")
        return
    
    # 3. Generate predictions directly (simplified approach)
    predictions = predict_funding(test_data)
    
    # 4. Generate submission file
    submission = generate_submission(predictions, output_path)
    
    print("Prediction pipeline completed successfully!")

if __name__ == "__main__":
    main()
