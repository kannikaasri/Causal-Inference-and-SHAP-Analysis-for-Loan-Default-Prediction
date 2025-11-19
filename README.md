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
- <a href="https://github.com/kannikaasri/Causal-Inference-and-SHAP-Analysis-for-Loan-Default-Prediction/blob/main/model_comparison.txt"> model_comparison </a> 

## SHAP Analysis

- Top global features influencing default risk:
  - employment_status_Unemployed
  - income
  - loan_amount
  - dti_ratio
- <a href="https://github.com/kannikaasri/Causal-Inference-and-SHAP-Analysis-for-Loan-Default-Prediction/blob/main/Screenshot%202025-11-18%20222157.png"> SHAP Analysis Screen shot</a>

### Five Highest Risk Applicants

Feature contributions for five highest-risk cases (details and plots in notebook/code).

## Causal Inference: Propensity Score Matching

- Treatment variable: High DTI (DTI ≥ 0.4)
- <a href="https://github.com/kannikaasri/Causal-Inference-and-SHAP-Analysis-for-Loan-Default-Prediction/blob/main/Screenshot%202025-11-18%20222236.png">  Propensity Score Screen shot</a>
- 951 matched pairs
- Estimated Average Treatment Effect (ATE): 0.4890
  - Interpretation: High-DTI applicants have a 48.9% higher default rate than matched low-DTI applicants.

## Policy Recommendation

Based on analysis:

- Loan applicants with DTI ≥ 0.4 are at significantly higher risk of default.
- It is recommended to tighten loan approval criteria for high-DTI applicants, and to target the most influential risk drivers found by SHAP (unemployment, low income, high loan amount) for closer scrutiny.


## References

- SHAP documentation: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
- Standard methodologies for propensity score matching and model evaluation.

## License

MIT License

