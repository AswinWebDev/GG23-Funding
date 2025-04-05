import pandas as pd

def fix_submission_format():
    """
    Ensures the submission file has the correct format with PROJECT_ID column.
    For MATURE BUILDERS projects that don't have IDs, leaves the field blank.
    """
    print("Fixing submission file format...")
    
    try:
        # Read the generated submission
        submission = pd.read_csv('submission.csv')
        
        # Read the original project data to get PROJECT_IDs
        projects = pd.read_csv('projects_Apr_1.csv')
        
        # Check if we already have the right format
        if 'PROJECT_ID' in submission.columns:
            print("Submission already has PROJECT_ID column")
            
            # Make sure MATURE BUILDERS projects have empty PROJECT_ID if needed
            mature_mask = submission['ROUND'] == 'MATURE BUILDERS'
            if mature_mask.any():
                print(f"Found {mature_mask.sum()} MATURE BUILDERS projects")
                
                # Check sample of the data
                print("\nSample of MATURE BUILDERS projects:")
                print(submission[mature_mask].head(3))
            
            # Check sample of other rounds
            other_mask = submission['ROUND'] != 'MATURE BUILDERS'
            if other_mask.any():
                print(f"\nSample of other round projects:")
                print(submission[other_mask].head(3))
        else:
            print("ERROR: Submission is missing PROJECT_ID column")
            
            # Create a proper submission with PROJECT_ID column
            print("Creating a properly formatted submission...")
            
            # Merge with original project data to get PROJECT_IDs
            fixed_submission = pd.merge(
                projects[['PROJECT_ID', 'PROJECT', 'ROUND']],
                submission[['PROJECT', 'ROUND', 'AMOUNT']],
                on=['PROJECT', 'ROUND'],
                how='left'
            )
            
            # Ensure proper order of columns
            fixed_submission = fixed_submission[['PROJECT_ID', 'PROJECT', 'ROUND', 'AMOUNT']]
            
            # Save fixed submission
            fixed_submission.to_csv('fixed_submission.csv', index=False)
            print("Fixed submission saved to fixed_submission.csv")
            
            # Show sample of fixed submission
            print("\nSample of fixed submission:")
            print(fixed_submission.head(5))
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fix_submission_format()
