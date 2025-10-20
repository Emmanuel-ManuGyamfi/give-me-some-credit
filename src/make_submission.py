import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from .features import load_train_features, load_test_features
from .config import Paths, RANDOM_SEED, SUBMISSION_COLS, ID_COL

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

def main(output_path: Path | None = None):
    X_train, y_train, _ = load_train_features()
    X_test, test_ids = load_test_features()

    # Align columns between train and test
    missing_in_test = set(X_train.columns) - set(X_test.columns)
    for c in missing_in_test:
        X_test[c] = 0
    extra_in_test = set(X_test.columns) - set(X_train.columns)
    if extra_in_test:
        X_test = X_test.drop(columns=list(extra_in_test))
    X_test = X_test[X_train.columns]

    print(f"[make_submission] Train: {X_train.shape}, Test: {X_test.shape}, Model: XGBoost")

    model = make_model()
    model.fit(X_train, y_train, verbose=False)

    test_prob = model.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({ID_COL: test_ids, "Probability": test_prob})[SUBMISSION_COLS]

    if output_path is None:
        output_path = Path("submission.csv")
    sub.to_csv(output_path, index=False)
    print(f"[make_submission] Saved {output_path.resolve()}")

    # Write metadata summary
    Paths.outputs.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "XGBoost",
        "params": XGB_PARAMS,
        "train_rows": int(X_train.shape[0]),
        "train_cols": int(X_train.shape[1]),
        "test_rows": int(X_test.shape[0]),
        "submission_rows": int(sub.shape[0]),
        "id_unique": int(sub[ID_COL].nunique()),
    }
    with open(Paths.outputs / "submission_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("[make_submission] Summary written to outputs/submission_summary.json")

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(out)
