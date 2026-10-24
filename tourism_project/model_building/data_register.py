# ============================================================
# DATA REGISTRATION AND VALIDATION
# ============================================================

import pandas as pd
import hashlib
from pathlib import Path
from schema import EXPECTED_COLUMNS, TARGET, ID_COLUMN

# Resolve the repository root
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "tourism_project/data/tourism.csv"


def register_dataset():

    print("=" * 55)
    print("VISIT WITH US - DATA REGISTRATION")
    print("=" * 55)

    # Step 1: Check whether the dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    # Step 2: Load dataset from the repository
    df = pd.read_csv(DATA_PATH)

    # Step 3: Validate expected columns
    missing_columns = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Step 4: Check customer ID integrity
    if df[ID_COLUMN].isna().any():
        raise ValueError("CustomerID contains missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate CustomerID values found.")

    # Step 5: Validate binary target
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("ProdTaken must contain only 0 and 1.")

    # Step 6: Dataset summary
    print("\nDATASET VALIDATION: PASSED")
    print("Total records:", len(df))
    print("Total columns:", len(df.columns))
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate records:", int(df.duplicated().sum()))

    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index())

    print("\nTarget percentage:")
    print(
        df[TARGET].value_counts(normalize=True)
        .sort_index().mul(100).round(2)
    )

    # Step 7: SHA-256 data fingerprint
    file_hash = hashlib.sha256(
        DATA_PATH.read_bytes()
    ).hexdigest()

    print("\nDataset SHA-256:", file_hash)
    print("\nData registration completed successfully.")

    return df


if __name__ == "__main__":
    register_dataset()
