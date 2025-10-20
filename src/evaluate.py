import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)

from .config import Paths

OUT_DIR = Paths.outputs

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _load_oof() -> pd.DataFrame:
    fp = OUT_DIR / "oof.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"Couldn't find {fp}. Run `make train` first so oof.csv is created."
        )
    df = pd.read_csv(fp)
    # Expected columns: ['oof_pred', 'target']
    if not {"oof_pred", "target"}.issubset(df.columns):
        raise ValueError("oof.csv must contain columns: oof_pred, target")
    return df

def _binary_preds(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    return (y_prob >= threshold).astype(int)

def main(threshold: float = 0.5):
    _ensure_dir(Path(OUT_DIR))
    df = _load_oof()
    y_true = df["target"].astype(int).values
    y_prob = df["oof_pred"].astype(float).values
    y_pred = _binary_preds(y_prob, threshold)

    # --- Scalar metrics ---
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": threshold,
        "positive_rate_pred": float(y_pred.mean()),
        "positive_rate_true": float(y_true.mean()),
    }

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # Save metrics JSON
    with open(OUT_DIR / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print a compact report to console
    print(json.dumps(metrics, indent=2))
    print("\nClassification report @ threshold", threshold)
    print(classification_report(y_true, y_pred, digits=4))

    # --- Plots ---
    # ROC
    fig_roc = plt.figure()
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve (OOF)")
    fig_roc.savefig(OUT_DIR / "roc_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig_roc)

    # Precision–Recall
    fig_pr = plt.figure()
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.title("Precision–Recall Curve (OOF)")
    fig_pr.savefig(OUT_DIR / "pr_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig_pr)

    # Confusion matrix heatmap (simple matplotlib)
    fig_cm = plt.figure()
    ax = plt.gca()
    im = ax.imshow(cm, aspect="equal")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    # annotate cells
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center")
    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    fig_cm.colorbar(im)
    fig_cm.savefig(OUT_DIR / "confusion_matrix.png", bbox_inches="tight", dpi=150)
    plt.close(fig_cm)

if __name__ == "__main__":
    # Change threshold here if you want (e.g., 0.23 for higher recall)
    main(threshold=0.23)
