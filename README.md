# GG23 Predictive Funding Challenge

## My Approach to Predicting Gitcoin Grants Round 23 Funding

This repository contains my solution to the Gitcoin Grants 23 (GG23) Predictive Funding Challenge. I've developed a model to predict how much funding each project will receive in the current round, with special focus on implementing Gitcoin's official funding metrics and the Cluster QF algorithm.

## Problem Overview

The challenge required me to predict funding outcomes for projects participating in Gitcoin Grants Round 23 across four categories:
- WEB3 INFRA ($200,000 matching pool)
- DEV TOOLING ($200,000 matching pool)
- DAPPS & APPS ($200,000 matching pool) 
- MATURE BUILDERS ($600,000 matching pool with no community contributions)

For three of the categories, I needed to predict both the matching pool allocation and community contributions, while for MATURE BUILDERS, only the matching pool allocation was required.

## My Data Analysis

### Historical Data

I began by exploring the historical Gitcoin funding data from "GG Allocation Since GG18.csv", which revealed several patterns:
- Different round categories show distinct funding patterns
- Strong correlation exists between contributor count and matching amounts
- Projects with consistent participation typically receive more funding
- User-facing applications generally attract more community contributions than infrastructure or tooling projects

### Challenges I Faced

I initially considered enhancing my model with additional data by scraping:
- GitHub metrics (stars, contributors, commit frequency)
- Social media presence and engagement
- Project maturity indicators

However, I encountered several challenges:
- The 36-hour submission window limited my ability to gather extensive data
- I faced rate limiting issues on various APIs
- Many projects had inconsistent data availability
- Web scraping raised potential ethical considerations

Given these constraints, I decided to focus on maximizing the predictive power of the provided historical data instead.

## My Modeling Approach

### Feature Engineering

I created several feature categories based on Gitcoin's official funding metrics:

1. **Ecosystem Growth (Allo GMV) metrics:**
   - Total matching funds received in previous rounds
   - Community round participation count
   - Participation consistency across rounds

2. **Donor Base Expansion & Loyalty metrics:**
   - Contributor count and growth
   - Donor retention (contributors who returned across rounds)
   - Average number of grantees supported per donor

3. **Builder Participation & Retention metrics:**
   - Active developers (estimated from contributor data)
   - Developer retention across rounds
   - New contributor onboarding rates

4. **Round-specific adjustments:**
   - Category-specific contribution multipliers
   - Base contribution rates tailored to round type
   - Different handling for returning vs. new projects

### My Prediction Methodology

I implemented a hybrid approach combining statistical analysis, domain knowledge, and Gitcoin's advanced funding principles:

1. **For the MATURE BUILDERS Round:**
   - I implemented Gitcoin's official funding metrics with a comprehensive scoring system:
     - Strong ecosystem impact scores for core infrastructure (e.g., Tor Project: 92, Ethers.io: 90)
     - Balanced metrics for DeFi & Analytics projects (e.g., DeFi Llama: 82, Revoke.cash: 79)
     - Community & Public Goods considerations (e.g., EthStaker: 77, Giveth: 74)
     - Developer tool specialized metrics (e.g., Rotki: 71)
   - I applied power scaling (exponent 1.2) to prioritize top projects
   - I allocated the $600,000 pool proportionally based on these weighted scores

2. **For the other three rounds:**
   - I implemented the Cluster QF algorithm as used by Gitcoin:
     - This matches funds based on the diversity of wallets contributing
     - Provides better resistance to contribution collusion
     - More accurately represents real community support
   - I applied matching caps consistent with Gitcoin's practice:
     - 10% matching cap for Web3/Dev Tooling
     - 5% matching cap for dApps
   - I modeled community contributions with round-specific boost multipliers

### Implementation

I implemented my model in Python using pandas and numpy, with these key components:

1. **Data preparation pipeline:**
   - Loading and cleaning historical data
   - Consistent column naming and PROJECT_ID handling
   - Mapping historical data to current projects
   - Calculating necessary metrics

2. **Feature extraction:**
   - Category-specific feature selection
   - Handling missing values
   - Feature normalization

3. **Prediction generation:**
   - Cluster QF implementation for accurate quadratic funding
   - Mature Builders ecosystem impact scoring
   - Community contribution estimation
   - Pool allocation validation

## My Results

My final model produced the following predictions:

1. **Total allocation by round:**
   DAPPS & APPS: 183 projects, $304,086.72
DEV TOOLING: 64 projects, $299,393.82
MATURE BUILDERS: 30 projects, $600,000.00
WEB3 INFRA: 60 projects, $300,876.21

2. **Distribution patterns:**
   - My model follows the power law distribution common in funding contexts
   - Top projects receive significantly larger allocations
   - There's a long tail of smaller allocations for newer projects
   - MATURE BUILDERS distribution reflects Gitcoin's focus on ecosystem impact metrics

## Limitations and Future Work

If I had more time and resources, I would improve my model by:

1. **Incorporating external data:**
   - GitHub metrics and development activity
   - Social media engagement indicators
   - Team experience metrics
   - Project impact assessments

2. **Using more advanced modeling techniques:**
   - Ensemble methods combining multiple prediction approaches
   - Time series analysis of funding trends
   - Network analysis of contributor relationships
   - Gradient boosting or neural networks

3. **Improving predictions for new projects:**
   - More sophisticated cold-start prediction methods
   - Better category-specific baseline estimates
   - Semantic similarity to previously funded projects

## Running the Code

To run my prediction model:

1. Ensure you have the required data files:
   - `GG Allocation Since GG18.csv` (historical data)
   - `projects_Apr_1.csv` (current projects list)

2. Run the prediction pipeline:
   ```
   python run_prediction.py
   ```

3. To include GitHub repository metrics (optional):
   ```
   $env:GITHUB_TOKEN = 'your_github_token_here'
   python run_prediction.py
   ```

4. The script will generate a `submission.csv` file with the predictions.

## Conclusion

My approach to the GG23 Predictive Funding Challenge combines statistical analysis with Gitcoin's official funding criteria. By implementing the Cluster QF algorithm and incorporating ecosystem impact metrics for MATURE BUILDERS, my model effectively captures the nuanced dynamics of Gitcoin Grants funding distribution across all project categories.
