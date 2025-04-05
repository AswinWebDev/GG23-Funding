import os
import subprocess
import time
from datetime import datetime

def main():
    """
    Main runner script that executes the full prediction pipeline:
    1. Researches projects (if GitHub token is available)
    2. Updates the model with research findings
    3. Runs the prediction
    """
    print(f"Gitcoin GG23 Prediction Pipeline Started at {datetime.now()}")
    
    # Check if GitHub token is available
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        print("GitHub token found. Will perform project research.")
        try:
            # Run the research script
            print("\n=== STEP 1: Researching Projects ===")
            subprocess.run(['python', 'research_projects.py'], check=True)
            print("Project research completed successfully.")
        except Exception as e:
            print(f"WARNING: Project research failed: {e}")
            print("Continuing with default project scores.")
    else:
        print("No GitHub token found. Skipping project research.")
        print("To enable research, set the GITHUB_TOKEN environment variable.")
        print("For example: $env:GITHUB_TOKEN = 'your_token_here'")
    
    # Run the prediction model
    print("\n=== STEP 2: Running Prediction Model ===")
    try:
        subprocess.run(['python', 'predict_funding.py'], check=True)
        print("Prediction model executed successfully.")
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        return
    
    # Verify output
    print("\n=== STEP 3: Verifying Results ===")
    if os.path.exists('submission.csv'):
        print("Submission file created successfully.")
        # Show sample of results
        try:
            with open('submission.csv', 'r') as f:
                lines = f.readlines()
                print("\nSample predictions:")
                print(lines[0], end='')  # Header
                
                # Show first few regular projects
                sample_count = 0
                for line in lines[1:]:
                    if 'MATURE BUILDERS' not in line:
                        print(line, end='')
                        sample_count += 1
                        if sample_count >= 3:
                            break
                
                # Show a few MATURE BUILDERS
                print("\nSample MATURE BUILDERS predictions:")
                mature_found = 0
                for line in lines:
                    if 'MATURE BUILDERS' in line:
                        print(line, end='')
                        mature_found += 1
                        if mature_found >= 3:
                            break
        except Exception as e:
            print(f"Error showing sample: {e}")
    else:
        print("WARNING: No submission file was created.")
    
    print(f"\nGitcoin GG23 Prediction Pipeline Completed at {datetime.now()}")
    print("\nNotes:")
    print("1. The model implements Cluster QF algorithm for accurate quadratic funding")
    print("2. Project IDs are used for matching where available")
    print("3. Matching caps are applied: 10% for Web3/Dev Tooling, 5% for dApps")
    print("4. MATURE BUILDERS projects use ecosystem impact scoring")
    
if __name__ == "__main__":
    main()
