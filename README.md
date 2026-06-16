# Customer Churn Prediction Dashboard

A machine learning and business intelligence project that predicts customer churn risk for an online betting platform. The project combines real sports odds data from The Odds API, simulated customer betting behavior, Python-based data analysis, SQL data extraction, scikit-learn model development, and a Power BI dashboard to help businesses identify high-risk customers and improve retention strategies.

## Project Overview

Customer churn is a major business risk for betting platforms that rely on recurring users and repeat wagers. This project analyzes customer activity patterns to predict which customers are likely to leave or become inactive.

The final output is an interactive Power BI dashboard that shows churn probability, customer risk segments, key churn factors, and recommended retention opportunities.

## Business Problem

Betting platforms often lose customers without identifying warning signs early enough. Manual reporting may show past churn, but it does not always help predict future churn.

This project solves that problem by:

* Identifying customers at high risk of leaving
* Finding behavioral patterns linked to churn
* Predicting churn probability using machine learning
* Presenting insights clearly through a Power BI dashboard
* Helping business teams prioritize retention actions

## Tech Stack

* **Python** — Data cleaning, EDA, feature engineering, and model training
* **SQL** — Data extraction, filtering, joins, and aggregation
* **scikit-learn** — Machine learning model development and evaluation
* **The Odds API** — Real sports odds and event data
* **Power BI** — Dashboard design and business insight visualization
* **Pandas / NumPy** — Data processing and transformation
* **Matplotlib / Seaborn** — Exploratory data visualization

## Key Features

* Customer churn prediction using machine learning
* Real sports odds data pulled from The Odds API (NFL, MLB, NHL, NBA, Soccer)
* Simulated customer base with realistic betting behavior and transaction history
* Exploratory Data Analysis to identify churn trends
* Feature engineering from customer behavior and transaction history
* Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
* Customer segmentation by churn risk level (High / Medium / Low)
* Power BI dashboard for business users
* Visual explanation of the main churn drivers

## Project Structure

```
Risk-Analysis-System/
├── main.py                    # Full pipeline runner
├── requirements.txt           # Python dependencies
├── .env.example               # API key template
├── src/
│   ├── config.py              # API config and simulation parameters
│   ├── data_collection.py     # Odds API fetch + customer simulation
│   ├── feature_engineering.py # Encoding, scaling, train/test split
│   └── model_training.py      # Train, evaluate, and export models
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory Data Analysis notebook
├── sql/
│   ├── 01_create_tables.sql   # Database schema
│   └── 02_data_extraction.sql # Analytical queries
├── data/
│   ├── raw/                   # API data, customers, transactions
│   └── processed/             # Features, predictions, charts
├── models/                    # Saved model artifacts
└── dashboard/                 # Power BI files
```

## Project Workflow

1. **Data Collection**

   * Pulled real sports events and odds from The Odds API (NFL, MLB, NHL, NBA, EPL).
   * Simulated 2,000 customers with realistic betting behavior over 180 days.
   * Generated 158,000+ betting transactions with stakes, odds, P&L, and balance tracking.

2. **Data Cleaning**

   * Removed duplicates and handled missing values.
   * Standardized column formats and corrected inconsistent records.
   * Prepared the dataset for analysis and model training.

3. **Exploratory Data Analysis**

   * Analyzed churn distribution across customer segments.
   * Identified trends in customer inactivity, transaction behavior, and account tenure.
   * Compared retained customers against churned customers across all key metrics.

4. **Feature Engineering**

   * Created features including customer tenure, average stake, days since last bet, bet frequency, win rate, total P&L, support ticket count, and promo usage.
   * Encoded categorical variables (sport preference, bet type, deposit method) and scaled numerical features.
   * Applied SMOTE oversampling to handle class imbalance.

5. **Model Development**

   * Trained and compared Logistic Regression, Random Forest, and Gradient Boosting.
   * Selected the best-performing model based on ROC-AUC, recall, and F1-score.

6. **Dashboard Development**

   * Exported model predictions and customer risk scores to CSV.
   * Built a Power BI dashboard to visualize churn risk, customer segments, and key churn indicators.

## Results

### Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.698 | 0.482 | 0.687 | 0.566 | **0.754** |
| Random Forest        | 0.708    | 0.491     | 0.496  | 0.494    | 0.731   |
| Gradient Boosting    | 0.713    | 0.500     | 0.391  | 0.439    | 0.716   |

**Best Model:** Logistic Regression (ROC-AUC: 0.754)

### Key Findings

* **2,000 customers** analyzed with **158,542 transactions** across 103 real sporting events
* **28.6% overall churn rate** — consistent with industry benchmarks for betting platforms
* **565 high-risk customers** identified, representing **$818,032 in total staked revenue at risk**
* Top churn drivers: total bets placed, days since last bet, total P&L, and support ticket count
* Customers with negative P&L and long inactivity periods are the most likely to churn
* High support ticket volume strongly correlates with churn

### Customer Risk Segmentation

| Segment     | Customers | % of Total |
|-------------|-----------|------------|
| Low Risk    | 626       | 31.3%      |
| Medium Risk | 809       | 40.5%      |
| High Risk   | 565       | 28.3%      |

## Dashboard Pages

### 1. Executive Summary

* Total customers analyzed
* Overall churn rate
* Number of high-risk customers
* Predicted revenue at risk
* Monthly churn trend

### 2. Customer Risk Segmentation

* High-risk, medium-risk, and low-risk customer groups
* Churn probability by customer segment
* Customer distribution by tenure and activity level

### 3. Churn Driver Analysis

* Key factors influencing churn
* Relationship between inactivity and churn
* Impact of P&L, bet frequency, and support interactions on churn

## How to Run

1. Clone the repository
2. Create a `.env` file with your API key:
   ```
   ODDS_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the full pipeline:
   ```
   python main.py
   ```
   Or run individual steps:
   ```
   python main.py --data-only    # Fetch data and build features
   python main.py --model-only   # Train models (requires existing data)
   ```
5. Open `notebooks/01_eda.ipynb` for the full exploratory analysis.
6. Import `data/processed/full_predictions_powerbi.csv` into Power BI for the dashboard.

## Data Source

* **The Odds API** (free tier) — Real-time sports odds from major bookmakers across NFL, NBA, MLB, NHL, and soccer leagues. Customer betting behavior is simulated on top of real event and odds data.
