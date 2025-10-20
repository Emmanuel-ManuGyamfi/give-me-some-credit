import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

from .config import Paths, RANDOM_SEED
from .utils import ensure_dirs
from .features import load_train_features

# --------------------------------------------------------------------------
# Tuned XGBoost hyperparameters
# --------------------------------------------------------------------------
XGB_PARAMS = dict(
    subsample=0.6,
    scale_pos_weight=1,
    reg_lambda=1,
    reg_alpha=0.5,
    n_estimators=800,
    min_child_weight=10,
    max_depth=4,
    learning_rate=0.01,
    gamma=0.2,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    tree_method="hist",
)

def make_model():
    return XGBClassifier(**XGB_PARAMS)

def main():
    ensure_dirs(Paths.outputs, Paths.data_processed)
    X, y, _ = load_train_features()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=float)
    scores = []

    for fold, (tr, va) in enumerate(cv.split(X, y), 1):
        model = make_model()
        model.fit(
            X.iloc[tr], y.iloc[tr],
            eval_set=[(X.iloc[va], y.iloc[va])],
            verbose=False,
        )
        prob = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y.iloc[va], prob)
        oof[va] = prob
        scores.append(auc)
        print(f"Fold {fold} ROC AUC: {auc:.6f}")

    mean_auc = float(np.mean(scores))
    std_auc = float(np.std(scores))
    print(f"\nCV ROC AUC: {mean_auc:.6f} ± {std_auc:.6f}")

    # Save out-of-fold predictions + metrics
    pd.DataFrame({"oof_pred": oof, "target": y}).to_csv(Paths.outputs / "oof.csv", index=False)
    with open(Paths.outputs / "cv_metrics.json", "w") as f:
        json.dump(
            {
                "roc_auc_mean": mean_auc,
                "roc_auc_std": std_auc,
                "model": "XGBoost",
                "params": XGB_PARAMS,
            },
            f,
            indent=2,
        )

if __name__ == "__main__":
    main()
