# Loan Default Prediction Using Machine Learning, SHAP Analysis, and Causal Inference

## Project Overview

This project predicts loan default risk using interpretable machine learning and causal inference. We use Logistic Regression and XGBoost, interpret predictions with SHAP, and estimate the impact of high Debt-to-Income Ratio (DTI) using propensity score matching.

## Objectives

- Predict loan default using financial and demographic features
- Compare Logistic Regression and XGBoost for performance
- Interpret global and local predictions using SHAP values
- Estimate causal effect of high DTI on default risk
- Provide actionable recommendations for lending policy

## Data and Preprocessing

- Dataset: Loan applicants with features — income, employment status, loan amount, DTI ratio, etc.
- Target variable: `default` (1 = defaulted, 0 = not)
- Key engineered feature: DTI ratio
- Data cleaned, missing values imputed, categorical variables encoded

## Model Training and Evaluation

| Model                | AUC   | F1    |
|----------------------|-------|-------|
| Logistic Regression  | 0.797 | 0.792 |
| XGBoost              | 0.761 | 0.693 |

**Best Model:** Logistic Regression

## SHAP Analysis

- Top global features influencing default risk:
  - employment_status_Unemployed
  - income
  - loan_amount
  - dti_ratio
- ![SHAP Summary Plot](images/shap_summary_plot_dti_0.4.png)

### Five Highest Risk Applicants

Feature contributions for five highest-risk cases (details and plots in notebook/code).

## Causal Inference: Propensity Score Matching

- Treatment variable: High DTI (DTI ≥ 0.4)
- ![Propensity Score Overlap](images/propensity_overlap_dti_0.4.png)
- 951 matched pairs
- Estimated Average Treatment Effect (ATE): 0.4890
  - Interpretation: High-DTI applicants have a 48.9% higher default rate than matched low-DTI applicants.

## Policy Recommendation

Based on analysis:

- Loan applicants with DTI ≥ 0.4 are at significantly higher risk of default.
- It is recommended to tighten loan approval criteria for high-DTI applicants, and to target the most influential risk drivers found by SHAP (unemployment, low income, high loan amount) for closer scrutiny.

## How to Reproduce



## References
SHAP documentation: https://shap.readthedocs.io/

Standard methodologies for propensity score matching and model evaluation.
1. Clone this repository.
2. Install requirements:
