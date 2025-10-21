# Give Me Some Credit — Probability of Default Model

This project builds a **credit risk scoring model** to predict the likelihood that a borrower will experience **financial distress within the next two years**.  
It is based on the [Kaggle "Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) competition dataset and demonstrates **end-to-end data science workflow** from data cleaning and feature engineering to model tuning and evaluation.

---

## Project Objectives
- Develop a **probability-of-default (PD)** model using credit data.  
- Apply **feature engineering** (handling outliers, imputations, data erros, and scaling).  
- Compare algorithms and select the champion model.  
- Communicate results effectively to **business stakeholders**.

---

## Tech Stack
- **Language:** Python 3.12  
- **Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn  
- **Environment:** Conda / venv  
- **Tools:** Jupyter Notebooks, GitHub, Kaggle  

---

## Dataset Overview
| Feature | Description |
|----------|-------------|
| SeriousDlqin2yrs | Target variable — 1 Person experienced 90 days past due delinquency or worse |
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits/>|
| DebtRatio | Debt payments, alimony, etc. divided by gross income |
| MonthlyIncome | Self-reported monthly income |
| NumberOfOpenCreditLinesAndLoans | Total number of open loans/credit lines |
| … | *(and other credit-behavioral predictors)* |

---

## 🧹 Data Preparation & Feature Engineering

- **Outlier Capping:**  
  - DebtRatio capped at 5  
  - Revolving utilization capped at 2× mean  
  - Late payment counts capped at 6  
- **Imputations:**  
  - Missing ages imputed with median  
  - Extreme or missing incomes imputed with mean  
- **Feature Creation:**  
  - `HighIncomeFlag`  
  - `TotalDelinquencies`  
  - `DependentsPerIncome`

🟦 *Example plot:*  
![Target Distribution](outputs/target_distribution.png)

---

## ⚙️ Model Development

Algorithm: **XGBoost Classifier**

| Hyperparameter | Value |
|----------------|--------|
| `n_estimators` | 800 |
| `learning_rate` | 0.01 |
| `max_depth` | 4 |
| `subsample` | 0.6 |
| `colsample_bytree` | 0.8 |
| `gamma` | 0.2 |
| `min_child_weight` | 10 |
| `scale_pos_weight` | 1 |
| `reg_alpha` | 0.5 |
| `reg_lambda` | 1 |

---

## 🧪 Model Evaluation

| Metric | Score |
|---------|-------|
| **ROC-AUC (CV)** | 0.8642 |
| **Kaggle Private Score** | 0.8661 |
| **Accuracy** | ~86% |
| **Precision (1)** | 0.62 |
| **Recall (1)** | 0.15 |
| **F1-score (1)** | 0.24 |

---

### 📈 Key Charts

- ROC Curve  
  ![ROC Curve](outputs/roc_curve.png)

- Precision-Recall Curve  
  ![PR Curve](outputs/pr_curve.png)

- Confusion Matrix  
  ![Confusion Matrix](outputs/confusion_matrix.png)

- Feature Importance  
  ![Feature Importance](outputs/feature_importance.png)

---

## 💡 Business Impact

This model allows financial institutions to:
- **Screen applicants** more accurately for credit risk  
- **Reduce bad debt rates** by targeting interventions  
- **Support fair lending** through interpretable variables  
- Simulate business impact:  
  > For every **100,000 applicants**, improved screening accuracy could reduce bad loans by ~8–10%, saving millions annually.

---

## 🧠 Learnings & Next Steps
- Handling class imbalance effectively using `scale_pos_weight` and proper thresholds.  
- Feature importance analysis to ensure fairness and interpretability.  
- Future work:  
  - Calibrated probability outputs  
  - SHAP-based interpretability  
  - Monitoring drift in production.

---

## 🏆 Competition Result
**Kaggle Score:** 0.85963 (Public) / 0.86608 (Private)  
![Kaggle Screenshot](outputs/kaggle_score.png)

---

## 📁 Repository Structure

MIT — see [LICENSE](LICENSE).
