# Give Me Some Credit — Probability of Default Model

This project builds a **credit risk scoring model** to predict the likelihood that a borrower will experience **financial distress within the next two years**.  
It is based on the [Kaggle "Give Me Some Credit"](https://www.kaggle.com/c/GiveMeSomeCredit) competition dataset and demonstrates **end-to-end data science workflow** from data cleaning and feature engineering to model tuning and evaluation.

---

## Project Objectives
- Develop a **probability-of-default (PD)** model using credit data.  
- Apply **feature engineering** (handling outliers, imputations, data errors, and scaling).  
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
| RevolvingUtilizationOfUnsecuredLines | Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits|
| Age| Age of borrower in years |
| NumberOfTime30-59DaysPastDueNotWorse | Number of times borrower has been 30-59 days past due but no worse in the last 2 years |
| NumberOfTime60-89DaysPastDueNotWorse | Number of times borrower has been 60-89 days past due but no worse in the last 2 years |
| NumberOfTimes90DaysLate | Number of times borrower has been 90 days or more past due |
| DebtRatio | Monthly debt payments, alimony,living costs divided by monthy gross income |
| MonthlyIncome | Self-reported monthly income |
| NumberOfOpenCreditLinesAndLoans | Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards) |
| NumberRealEstateLoansOrLines | Number of mortgage and real estate loans including home equity lines of credit |
| NumberOfDependents | Number of dependents in family excluding themselves (spouse, children etc.) |
---

## Data Preparation & Feature Engineering

- **Outlier Capping:**  
  - `DebtRatio` capped at 5  
  - `RevolvingUtilizationOfUnsecuredLines` capped at 2 
  - `NumberOfTime30-59DaysPastDueNotWorse` capped at 6
  - `NumberOfTime60-89DaysPastDueNotWorse` capped at 6
  - `NumberOfTimes90DaysLate` capped at 6
  - `NumberOfDependents` capped at 5
  - `NumberOfOpenCreditLinesAndLoans` capped at 30
  - `NumberRealEstateLoansOrLines` capped at 5
- **Imputations:**  
  - Replaced implausible values (e.g., over 100 years) with the median age of valid applicants (18–100) 
  - Extreme monthly incomes imputed with mean of high earners
- **Feature Creation:**  `  
  - `TotalDelinquencies`  
  - `DependentsPerIncome`

🟦 *Example plot:*  
<Figure size 500x500 with 1 Axes><img width="404" height="427" alt="image" src="https://github.com/user-attachments/assets/66f9bdd3-90c1-4d42-96d7-508818a31d44" />

## Model Evaluation

The models were evaluated using multiple metrics to capture both discrimination and balance performance.  
Given the **high class imbalance** (only ~6.7% defaults), precision and recall are more informative than accuracy alone.

| Model | Precision | Recall | F1 | Accuracy | ROC-AUC |
|--------|------------|--------|------------|----------|----------|
| **Random Forest (CV)** | 0.615688 | 0.144719 | 0.233992 | 0.936533 | 0.862698 |
| **Random Forest (Test)** | 0.619932 | 0.147330 | 0.238080 | 0.937105 | 0.860421 |
| **XGBoost (CV)** | 0.601807 | 0.166800 | 0.260878 | 0.936676 | **0.864169** |
| **XGBoost (Test)** | 0.614815 | 0.166600 | **0.262160** | **0.937453** | 0.862600 |

**AUC-ROC & PR-AUC**
- XGBoost achieved a slightly higher ROC-AUC (0.863) compared to Random Forest (0.860), indicating stronger overall ranking ability.
- The Precision–Recall AUC (0.405 vs 0.398) also favors XGBoost, confirming better performance on the imbalanced target (defaults).
**Recall & F1-score** 
- XGBoost captures more true defaulters (recall = 0.167) than Random Forest (recall = 0.147), while maintaining a similar precision.
- This leads to a higher F1-score, meaning XGBoost provides a better balance between precision and recall.
**Accuracy**
- Accuracy remains similar across models (~93.7%), but in imbalanced problems, accuracy is less meaningful than AUC or recall.
**Key Takeaways:**
- High accuracy reflects class imbalance — thus ROC-AUC and PR-AUC were the main model selection metrics.
- This level of performance indicates strong predictive separation between low- and high-risk applicants.

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
