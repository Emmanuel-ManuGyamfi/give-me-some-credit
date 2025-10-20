"""
Configuration file for Give Me Some Credit project.
Holds constant variables and directory paths used across the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

# ------------------------------------------------------------------------------
# Project-wide constants
# ------------------------------------------------------------------------------

# Target variable in the training data
TARGET = "SeriousDlqin2yrs"

# Unique identifier column
ID_COL = "Id"

# Random seed for reproducibility
RANDOM_SEED = 42


# ------------------------------------------------------------------------------
# 📁 Directory paths
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """
    Central location for directory paths.
    These are used throughout src/ for reading/writing data and outputs.
    """
    # Raw Kaggle CSVs
    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")

    # Model outputs, charts, and submission files
    outputs: Path = Path("outputs")

# ------------------------------------------------------------------------------
# Submission format
# ------------------------------------------------------------------------------

# Columns for final Kaggle submission file
SUBMISSION_COLS = [ID_COL, "Probability"]
