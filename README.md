# Give Me Some Credit — Kaggle (GMSC)

Reproducible code and write-up for the **Give Me Some Credit** Kaggle competition.

This repo contains a clean pipeline to:
- create features
- train models with cross-validation
- evaluate offline metrics
- generate a Kaggle-ready `submission.csv`

> **Status:** Complete first submission uploaded on 2025-10-20.

## Quickstart

```bash
# 1) Clone and enter
git clone <YOUR_REPO_URL> give-me-some-credit
cd give-me-some-credit

# 2) Create environment (pick one)
conda env create -f environment.yml && conda activate gmsc
# OR
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 3) Put raw data here (ignored by git)
# data/raw/cs-training.csv
# data/raw/cs-test.csv

# (Optional) Use Kaggle API to download (requires kaggle.json configured)
make data

# 4) Reproduce pipeline
make train
make evaluate
make submit
```

## Repository structure

```
.
├── README.md
├── LICENSE
├── environment.yml
├── requirements.txt
├── Makefile
├── .gitignore
├── notebooks/
│   ├── 01-eda.ipynb
│   └── 02-model-experiments.ipynb
├── src/
│   ├── config.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── make_submission.py
│   └── utils.py
└── data/
    ├── raw/        # put Kaggle CSVs here (gitignored)
    ├── interim/    # intermediate files (gitignored)
    └── processed/  # model-ready parquet/csv (gitignored)
```

## Data

- Competition: *Give Me Some Credit*
- Columns like `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio`, etc.
- **Do not commit raw data** — this repo ignores `data/` by default.

## Approach (short)
- Robust preprocessing: missing values, outlier handling, and flags for imputations (e.g., high income flag).
- Feature set: utilization, delinquencies, debt ratios, engineered interactions; SHAP/importance for interpretability.
- Models: baseline Logistic Regression; tree-based (RandomForest / XGBoost) tuned by CV.
- Validation: stratified K-fold across full training set with fixed random seed.
- Metric: ROC AUC (primary), plus PR AUC, F1, and recall.
- Deliverables: `submission.csv` with `Id` and `Probability` columns.

## Reproducibility
- Fixed `RANDOM_SEED=42`.
- `environment.yml` / `requirements.txt` lock key versions.
- Deterministic scikit-learn settings where applicable.

## Results
- Public LB score: *add your score here*  
- Local CV ROC AUC: *add your mean ± std here*
- Notes on any generalization gap between CV and public scoreboard.

## How to rerun with your code
Drop your finished feature engineering into `src/features.py` and your trained model choice into `src/train.py`. The CLI entry points are already wired via the Makefile.

## Model Card (brief)
- **Intended use:** credit distress prediction over 2 years.
- **Limitations:** dataset shift risk, sensitive attribute handling, potential fairness concerns.
- **Ethics:** apply responsibly; avoid individual-level adverse action explanations without rigorous review.

## Citation
If you reference this repo, please cite via `CITATION.cff` or link to this page.

## License
MIT — see [LICENSE](LICENSE).
