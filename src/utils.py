from __future__ import annotations
import json
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------
def ensure_dirs(*dirs: Iterable[Path | str]) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Reproducibility & timing
# ---------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Best-effort seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        # torch not installed or GPU not present — that's fine
        pass


@contextmanager
def timeit(label: str):
    """Context manager to time a code block."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[timeit] {label}: {dt:.3f}s")


# ---------------------------------------------------------------------
# Lightweight I/O helpers
# ---------------------------------------------------------------------
def save_json(obj: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_figure(fig, path: Path | str, dpi: int = 150, bbox_inches: str = "tight") -> None:
    """Save a matplotlib figure safely (no import if unused)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)


# ---------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------
def print_df_info(df: pd.DataFrame, name: str = "df") -> None:
    """Compact schema + nulls overview."""
    rows, cols = df.shape
    nulls = df.isna().sum().sort_values(ascending=False)
    print(f"[{name}] shape={rows}x{cols}")
    if not nulls.empty:
        print(f"[{name}] top nulls:\n{nulls.head(10)}")


def reduce_mem_usage(df: pd.DataFrame, use_float16: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:

    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    dtypes_map: Dict[str, str] = {}

    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
            dtypes_map[col] = str(df[col].dtype)
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
            if use_float16 and str(df[col].dtype) == "float32":
                # optional float16 step — only if you know your model tolerates it
                try:
                    df[col] = df[col].astype("float16")
                    dtypes_map[col] = "float16"
                except Exception:
                    dtypes_map[col] = "float32"
            else:
                dtypes_map[col] = str(df[col].dtype)

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"[reduce_mem_usage] {start_mem:.2f} MB → {end_mem:.2f} MB")
    return df, dtypes_map


def safe_read_csv(path: Path | str, **kwargs) -> pd.DataFrame:
    """
    Robust CSV reader with sensible defaults for large numeric CSVs.
    You can override kwargs as needed.
    """
    defaults = dict(encoding="utf-8", low_memory=False)
    defaults.update(kwargs)
    return pd.read_csv(path, **defaults)
