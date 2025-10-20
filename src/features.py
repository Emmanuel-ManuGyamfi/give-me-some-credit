"""
Feature engineering module for Give Me Some Credit.

This file transforms raw Kaggle CSVs (train/test) into cleaned feature sets
ready for model training and submission generation.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .config import Paths, TARGET, ID_COL
from .utils import ensure_dirs


@dataclass
class FeatureBuilder:
    """Fit/transform pipeline for consistent feature engineering."""

    medians_: Dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df_train: pd.DataFrame):
        """Fit any statistics (like medians) on train data."""
        df = df_train.copy()

        # Standardize ID column
        if ID_COL not in df.columns:
            if "Unnamed: 0" in df.columns:
                df.rename(columns={"Unnamed: 0": ID_COL}, inplace=True)
            else:
                df.insert(0, ID_COL, np.arange(1, len(df) + 1))

        # Learn medians for imputation
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            if c != TARGET:
                self.medians_[c] = float(df[c].median(skipna=True))

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Apply the full feature engineering pipeline."""
        assert self.fitted_, "Call .fit() before .transform()."
        out = df.copy()

        # Standardize ID
        if ID_COL not in out.columns:
            if "Unnamed: 0" in out.columns:
                out.rename(columns={"Unnamed: 0": ID_COL}, inplace=True)
            else:
                out.insert(0, ID_COL, np.arange(1, len(out) + 1))

        # FEATURE ENGINEERING

        # ---- Age Outlier Handling ----
        valid_age_min = 18
        valid_age_max = 100
        out["Age_Outlier"] = ((out["age"] < valid_age_min) | (out["age"] > valid_age_max)).astype(int)
        out.loc[out["Age_Outlier"] == 1, "age"] = np.nan
        age_median = out["age"].median() if is_train else self.medians_.get("age", out["age"].median())
        out["age_imputed"] = out["age"].fillna(age_median)

        # ---- Monthly Income Outliers ----
        high_income = out[out["MonthlyIncome"] > 50000]
        if not high_income.empty:
            p95_hi = high_income["MonthlyIncome"].quantile(0.95)
        else:
            p95_hi = out["MonthlyIncome"].quantile(0.95)

        out["ExtremeIncomeFlag"] = (out["MonthlyIncome"] > p95_hi).astype(int)
        impute_value = out["MonthlyIncome"].mean() if is_train else self.medians_.get("MonthlyIncome", out["MonthlyIncome"].mean())
        out["Missing_MonthlyIncome_Imputed"] = np.where(
            out["ExtremeIncomeFlag"] == 1, impute_value, out["MonthlyIncome"]
        )

        # ---- Revolving Utilization ----
        mean_2 = out.loc[out["RevolvingUtilizationOfUnsecuredLines"] <= 2, "RevolvingUtilizationOfUnsecuredLines"].mean()
        out["RevolvingUtilization_cleaned"] = np.where(
            out["RevolvingUtilizationOfUnsecuredLines"] > 2,
            mean_2,
            out["RevolvingUtilizationOfUnsecuredLines"],
        )

        # ---- Debt Ratio ----
        out["DebtRatio_cleaned"] = out["DebtRatio"].clip(upper=5)

        # ---- Delinquency Caps ----
        out["NumberOfTimes90DaysLate"] = out["NumberOfTimes90DaysLate"].clip(upper=6)
        out["NumberOfTime30-59DaysPastDueNotWorse"] = out["NumberOfTime30-59DaysPastDueNotWorse"].clip(upper=6)
        out["NumberOfOpenCreditLinesAndLoans_cleaned"] = out["NumberOfOpenCreditLinesAndLoans"].clip(upper=30)
        out["NumberRealEstateLoansOrLines_cleaned"] = out["NumberRealEstateLoansOrLines"].clip(upper=5)
        out["NumberOfDependentsMissing_cleaned"] = out["NumberOfDependents"].clip(upper=5)

        # ---- Combined delinquency score ----
        out["TotalDelinquencies"] = (
            out["NumberOfTime30-59DaysPastDueNotWorse"]
            + out["NumberOfTime60-89DaysPastDueNotWorse"]
            + out["NumberOfTimes90DaysLate"]
        )

        # ---- Dependents per Income ----
        out["DependentsPerIncome"] = out["NumberOfDependents"] / (out["MonthlyIncome"] + 1)

        # ---- Final feature selection ----
        keep_cols = [
            ID_COL,
            "SeriousDlqin2yrs",
            "age_imputed",
            "Missing_MonthlyIncome_Imputed",
            "RevolvingUtilization_cleaned",
            "DebtRatio_cleaned",
            "NumberOfOpenCreditLinesAndLoans_cleaned",
            "NumberRealEstateLoansOrLines_cleaned",
            "NumberOfDependentsMissing_cleaned",
            "DependentsPerIncome",
            "TotalDelinquencies",
        ]

        # Keep only columns that exist
        keep_cols = [c for c in keep_cols if c in out.columns]
        out = out[keep_cols]

        return out


# ---------------- Public Helper Functions ---------------- #

def _load_raw(split: str) -> pd.DataFrame:
    """Read raw CSVs placed in data/raw/."""
    fp = Paths.data_raw / ("cs-training.csv" if split == "train" else "cs-test.csv")
    return pd.read_csv(fp)


def load_train_features():
    """Load, clean, and return training features (X, y, ids)."""
    ensure_dirs(Paths.outputs, Paths.data_processed)
    df_tr = _load_raw("train")

    fb = FeatureBuilder().fit(df_tr)
    df_feat = fb.transform(df_tr, is_train=True)

    y = df_feat[TARGET].astype(int)
    ids = df_feat[ID_COL]
    X = df_feat.drop(columns=[ID_COL, TARGET])
    return X, y, ids


def load_test_features():
    """Load, clean, and return test features (X, ids)."""
    df_tr = _load_raw("train")
    df_te = _load_raw("test")

    fb = FeatureBuilder().fit(df_tr)
    df_te_feat = fb.transform(df_te, is_train=False)

    ids = df_te_feat[ID_COL]
    X_te = df_te_feat.drop(columns=[ID_COL])
    return X_te, ids
