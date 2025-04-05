import pandas as pd
import numpy as np
import requests
import json
import time
import re
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor

# GitHub API token (you need to set this as an environment variable)
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')

# Base URLs for APIs
GITHUB_API = 'https://api.github.com'
ETHERSCAN_API = 'https://api.etherscan.io/api'
DUNE_API = 'https://api.dune.com/api/v1'

class ProjectResearcher:
    """Class to research project metrics and ecosystem impact"""
    
    def __init__(self, projects_file='projects_Apr_1.csv', output_file='project_metrics.csv'):
        self.projects_file = projects_file
        self.output_file = output_file
        self.projects_df = None
        self.metrics_df = None
        self.headers = {
            'Authorization': f'token {GITHUB_TOKEN}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
    def load_projects(self):
        """Load projects from CSV file"""
        self.projects_df = pd.read_csv(self.projects_file)
        print(f"Loaded {len(self.projects_df)} projects")
        
        # Filter for MATURE BUILDERS
        self.mature_builders = self.projects_df[self.projects_df['ROUND'] == 'MATURE BUILDERS']
        print(f"Found {len(self.mature_builders)} MATURE BUILDERS projects")
        
        # Initialize metrics dataframe
        self.metrics_df = pd.DataFrame({
            'PROJECT': self.mature_builders['PROJECT'],
            'github_stars': np.nan,
            'github_forks': np.nan,
            'github_contributors': np.nan,
            'github_commits': np.nan,
            'github_last_activity': np.nan,
            'twitter_followers': np.nan,
            'ecosystem_mentions': np.nan,
            'composite_score': np.nan
        })
        
    def find_github_repo(self, project_name):
        """Find the GitHub repository for a project"""
        # Try different search strategies
        search_terms = [
            project_name,
            project_name.replace('-', ''),
            project_name.replace('-', ' ')
        ]
        
        for term in search_terms:
            try:
                url = f"{GITHUB_API}/search/repositories?q={term}+in:name&sort=stars&order=desc"
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    if data['total_count'] > 0:
                        # Return the most relevant repository
                        return data['items'][0]
                time.sleep(1)  # Rate limit protection
            except Exception as e:
                print(f"Error searching for {project_name}: {e}")
                
        return None
    
    def get_github_metrics(self, repo):
        """Extract GitHub metrics from repo data"""
        if not repo:
            return {
                'github_stars': np.nan,
                'github_forks': np.nan,
                'github_contributors': np.nan,
                'github_commits': np.nan,
                'github_last_activity': np.nan
            }
            
        metrics = {
            'github_stars': repo.get('stargazers_count', 0),
            'github_forks': repo.get('forks_count', 0),
            'github_last_activity': repo.get('updated_at', '')
        }
        
        # Get contributors count
        try:
            contributors_url = repo['contributors_url']
            response = requests.get(contributors_url, headers=self.headers)
            if response.status_code == 200:
                # GitHub may not return all contributors in one page
                contributors = response.json()
                metrics['github_contributors'] = len(contributors)
            time.sleep(1)  # Rate limit protection
        except:
            metrics['github_contributors'] = np.nan
            
        # Get commit count (approximation)
        try:
            commits_url = f"{repo['url']}/commits"
            response = requests.get(commits_url, headers=self.headers)
            if response.status_code == 200:
                # GitHub doesn't directly give total commits, but we can estimate
                link_header = response.headers.get('Link', '')
                if 'rel="last"' in link_header:
                    last_page = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if last_page:
                        metrics['github_commits'] = int(last_page.group(1)) * 30  # Approximate
                else:
                    metrics['github_commits'] = len(response.json())
            time.sleep(1)  # Rate limit protection
        except:
            metrics['github_commits'] = np.nan
            
        return metrics
    
    def get_twitter_metrics(self, project_name):
        """Find Twitter follower count (note: requires Twitter API access)"""
        # This is a placeholder - Twitter API requires authentication
        # In a real implementation, you would use the Twitter API
        return {'twitter_followers': np.nan}
    
    def get_ecosystem_mentions(self, project_name):
        """Find mentions in blockchain ecosystem"""
        # This would search for project mentions in key ecosystem sources
        # Placeholder for a real implementation
        return {'ecosystem_mentions': np.nan}
    
    def research_project(self, project_name):
        """Research a single project and return metrics"""
        print(f"Researching {project_name}...")
        
        # Find GitHub repository
        repo = self.find_github_repo(project_name)
        
        # Get metrics
        metrics = {}
        metrics.update(self.get_github_metrics(repo))
        metrics.update(self.get_twitter_metrics(project_name))
        metrics.update(self.get_ecosystem_mentions(project_name))
        
        return {
            'PROJECT': project_name,
            **metrics
        }
    
    def research_all_projects(self):
        """Research all MATURE BUILDERS projects"""
        print("Starting research for all projects...")
        
        results = []
        project_names = self.mature_builders['PROJECT'].tolist()
        
        # Use ThreadPoolExecutor to parallelize research
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(self.research_project, project_names))
        
        # Update metrics dataframe
        for result in results:
            project = result['PROJECT']
            for key, value in result.items():
                if key != 'PROJECT':
                    self.metrics_df.loc[self.metrics_df['PROJECT'] == project, key] = value
                    
        # Calculate composite score
        self.calculate_composite_scores()
        
        # Save to CSV
        self.metrics_df.to_csv(self.output_file, index=False)
        print(f"Research complete. Results saved to {self.output_file}")
        
        return self.metrics_df
    
    def calculate_composite_scores(self):
        """Calculate composite scores based on available metrics"""
        # Normalize each metric to 0-100 scale
        for column in ['github_stars', 'github_forks', 'github_contributors', 'github_commits']:
            if self.metrics_df[column].notna().any():
                max_val = self.metrics_df[column].max()
                if max_val > 0:
                    self.metrics_df[f'{column}_normalized'] = self.metrics_df[column] / max_val * 100
                else:
                    self.metrics_df[f'{column}_normalized'] = 0
            else:
                self.metrics_df[f'{column}_normalized'] = 50  # Default if no data
        
        # Calculate composite score
        # This is a weighted average of available normalized metrics
        self.metrics_df['composite_score'] = (
            self.metrics_df['github_stars_normalized'] * 0.4 +
            self.metrics_df['github_contributors_normalized'] * 0.3 +
            self.metrics_df['github_commits_normalized'] * 0.2 +
            self.metrics_df['github_forks_normalized'] * 0.1
        )
        
        # Fill NaN with reasonable default
        self.metrics_df['composite_score'] = self.metrics_df['composite_score'].fillna(50)
        
        # Add a small random component (±10%)
        self.metrics_df['composite_score'] = self.metrics_df['composite_score'] * np.random.uniform(0.9, 1.1, size=len(self.metrics_df))
        
        # Ensure all scores are within 0-100
        self.metrics_df['composite_score'] = self.metrics_df['composite_score'].clip(0, 100)
        
    def validate_with_historical_data(self, historical_data_path='GG Allocation Since GG18.csv'):
        """Validate predictions against historical data"""
        try:
            historical_df = pd.read_csv(historical_data_path)
            
            # Check correlation between our scores and historical funding amounts
            merged_df = pd.merge(
                self.metrics_df,
                historical_df[['PROJECT', 'AMOUNT']],
                on='PROJECT',
                how='inner'
            )
            
            if len(merged_df) > 5:  # Only if we have enough overlap
                correlation = merged_df['composite_score'].corr(merged_df['AMOUNT'])
                print(f"Correlation with historical funding: {correlation:.4f}")
                return correlation
            else:
                print("Not enough overlapping projects to calculate correlation")
                return None
        except Exception as e:
            print(f"Error validating with historical data: {e}")
            return None

def update_prediction_model():
    """Update the prediction model with researched metrics"""
    # Load project metrics
    if os.path.exists('project_metrics.csv'):
        metrics_df = pd.read_csv('project_metrics.csv')
        
        # Read the prediction script
        with open('predict_funding.py', 'r') as f:
            script = f.read()
            
        # Update the MATURE BUILDERS scores in the script
        project_metrics = {}
        for idx, row in metrics_df.iterrows():
            project = row['PROJECT'].lower().strip()
            score = row['composite_score']
            if not pd.isna(score):
                project_metrics[project] = int(score)
        
        # Convert to Python code
        metrics_code = "known_projects = {\n"
        for project, score in project_metrics.items():
            metrics_code += f"    '{project}': {score},  # Updated from GitHub metrics\n"
        metrics_code += "}"
        
        # Replace the placeholder section
        pattern = r"known_projects = \{[^}]+\}"
        updated_script = re.sub(pattern, metrics_code, script)
        
        # Write back to the prediction script
        with open('predict_funding.py', 'w') as f:
            f.write(updated_script)
            
        print("Updated prediction model with researched metrics")
        return True
    else:
        print("No metrics file found. Run research first.")
        return False

def main():
    """Main entry point for project research"""
    print(f"Starting project research at {datetime.now()}")
    
    # Create researcher
    researcher = ProjectResearcher()
    
    # Load projects
    researcher.load_projects()
    
    # Research all projects
    researcher.research_all_projects()
    
    # Update prediction model
    update_prediction_model()
    
    print(f"Research completed at {datetime.now()}")

if __name__ == "__main__":
    main()
