GG23 Predictive Funding Challenge
Predicting Gitcoin Grants Round 23 Funding
This repository contains a solution to the Gitcoin Grants 23 (GG23) Predictive Funding Challenge. The model predicts funding outcomes for projects participating in Gitcoin Grants Round 23, implementing Gitcoin's official funding metrics and the Cluster QF algorithm.
Problem Overview
The challenge requires predicting funding outcomes for projects across four categories:

WEB3 INFRA ($200,000 matching pool)
DEV TOOLING ($200,000 matching pool)
DAPPS & APPS ($200,000 matching pool)
MATURE BUILDERS ($600,000 matching pool with no community contributions)

For three categories, the prediction includes both matching pool allocation and community contributions, while for MATURE BUILDERS, only the matching pool allocation is required.
Data Analysis
Analysis of historical Gitcoin funding data from "GG Allocation Since GG18.csv" revealed:

Different round categories show distinct funding patterns
Strong correlation between contributor count and matching amounts
Projects with consistent participation typically receive more funding
User-facing applications generally attract more community contributions

Modeling Approach
Feature Engineering
Features based on Gitcoin's funding metrics include:

Ecosystem Growth metrics:

Matching funds from previous rounds
Community round participation count
Participation consistency

Donor Base metrics:

Contributor count and growth
Donor retention across rounds
Average grantees per donor

Builder Participation metrics:

Active developers estimation
Developer retention
New contributor rates

Prediction Methodology
The approach combines statistical analysis with Gitcoin's funding principles:

MATURE BUILDERS Round:

Comprehensive scoring system based on ecosystem impact
Power scaling to prioritize top projects
Proportional allocation of the $600,000 pool

Other three rounds:

Cluster QF algorithm implementation
Matching caps (10% for Web3/Dev Tooling, 5% for dApps)
Round-specific community contribution modeling

Results
The model produced these predictions:

DAPPS & APPS: 183 projects, $304,086.72
DEV TOOLING: 64 projects, $299,393.82
MATURE BUILDERS: 30 projects, $600,000.00
WEB3 INFRA: 60 projects, $300,876.21

The distribution follows the power law common in funding contexts, with top projects receiving larger allocations and a long tail of smaller allocations for newer projects.
Running the Code
To run the prediction model:
Copypython run_prediction.py
To include GitHub repository metrics (optional):
Copy$env:GITHUB_TOKEN = 'your_github_token_here'
python run_prediction.py
The script generates a submission.csv file with the predictions.
