# Customer Churn Prediction Dashboard

A machine learning and business intelligence project that predicts customer churn risk using customer behavior and transaction data. The project combines Python-based data analysis, SQL data extraction, scikit-learn model development, and a Power BI dashboard to help businesses identify high-risk customers and improve retention strategies.

## Project Overview

Customer churn is a major business risk for companies that rely on recurring users, subscriptions, or repeat transactions. This project analyzes customer activity patterns to predict which customers are likely to leave or become inactive.

The final output is an interactive Power BI dashboard that shows churn probability, customer risk segments, key churn factors, and recommended retention opportunities.

## Business Problem

Businesses often lose customers without identifying warning signs early enough. Manual reporting may show past churn, but it does not always help predict future churn.

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
* **Power BI** — Dashboard design and business insight visualization
* **Pandas / NumPy** — Data processing and transformation
* **Matplotlib / Seaborn** — Exploratory data visualization

## Key Features

* Customer churn prediction using machine learning
* Exploratory Data Analysis to identify churn trends
* Feature engineering from customer behavior and transaction history
* Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
* Customer segmentation by churn risk level
* Power BI dashboard for business users
* Visual explanation of the main churn drivers

## Project Workflow

1. **Data Collection**

   * Imported customer and transaction data from a SQL database.
   * Extracted relevant fields such as customer activity, purchase frequency, tenure, support interactions, and account status.

2. **Data Cleaning**

   * Removed duplicates and handled missing values.
   * Standardized column formats and corrected inconsistent records.
   * Prepared the dataset for analysis and model training.

3. **Exploratory Data Analysis**

   * Analyzed churn distribution across customer segments.
   * Identified trends in customer inactivity, transaction behavior, and account tenure.
   * Compared retained customers against churned customers.

4. **Feature Engineering**

   * Created new features such as customer tenure, average monthly spend, days since last activity, transaction frequency, and support issue count.
   * Encoded categorical variables and scaled numerical features for model training.

5. **Model Development**

   * Trained a churn prediction model using scikit-learn.
   * Compared multiple models such as Logistic Regression, Random Forest, and Gradient Boosting.
   * Selected the best-performing model based on recall, F1-score, and ROC-AUC.

6. **Dashboard Development**

   * Exported model predictions and customer risk scores.
   * Built a Power BI dashboard to visualize churn risk, customer segments, and key churn indicators.
   * Designed the dashboard for business decision-making and retention planning.

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
* Impact
