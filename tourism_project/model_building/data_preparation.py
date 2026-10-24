# ============================================================
# VISIT WITH US - MLOPS PROJECT
# SECTION 2: DATA PREPARATION
# ============================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from schema import (
    TARGET,
    ID_COLUMN,
    EXPECTED_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    FEATURE_COLUMNS,
    POST_CONTACT_COLUMNS,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "tourism_project/data/tourism.csv"
OUTPUT_DIR = ROOT / "tourism_project/outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def file_sha256(path):
    """Generate a SHA-256 fingerprint for reproducibility."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data():

    print("=" * 60)
    print("VISIT WITH US | DATA PREPARATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Step 1: Load data directly from the repository
    # --------------------------------------------------------

    if not SOURCE.is_file():
        raise FileNotFoundError(
            f"Missing registered dataset: {SOURCE}"
        )

    raw = pd.read_csv(SOURCE)

    print("\nDataset loaded successfully.")

    # --------------------------------------------------------
    # Step 2: Validate the source data
    # --------------------------------------------------------

    missing = sorted(
        set(EXPECTED_COLUMNS) - set(raw.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if (
        raw[ID_COLUMN].isna().any()
        or raw[ID_COLUMN].duplicated().any()
    ):
        raise ValueError(
            "CustomerID must be non-null and unique."
        )

    if (
        raw[TARGET].isna().any()
        or not raw[TARGET].isin([0, 1]).all()
    ):
        raise ValueError(
            "ProdTaken must be a non-null binary target."
        )

    # --------------------------------------------------------
    # Step 3: Remove unnecessary and post-contact columns
    # --------------------------------------------------------

    # Retain only the features defined in schema.py.
    # This excludes IDs and information unavailable before
    # contacting the customer.

    clean = raw[FEATURE_COLUMNS + [TARGET]].copy()

    clean[TARGET] = clean[TARGET].astype(int)

    # --------------------------------------------------------
    # Step 4: Clean categorical features
    # --------------------------------------------------------

    for column in CATEGORICAL_COLUMNS:

        values = clean[column].astype("string").str.strip()

        # Represent blank text as missing information.
        values = values.replace("", pd.NA)

        clean[column] = (
            values.astype(object)
            .where(values.notna(), np.nan)
        )

    # Correct inconsistent category spellings.
    clean["Gender"] = clean["Gender"].replace({
        "Fe Male": "Female"
    })

    clean["Occupation"] = clean["Occupation"].replace({
        "Free Lancer": "Freelancer"
    })

    # --------------------------------------------------------
    # Step 5: Standardize numeric data types
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        clean[column] = pd.to_numeric(
            clean[column],
            errors="coerce"
        )

    # Do NOT perform global missing-value imputation here.
    # Imputers and encoders will be fitted only on training
    # folds in the model-building pipeline.

    # --------------------------------------------------------
    # Step 6: Stratified train/test split
    # --------------------------------------------------------

    train, test = train_test_split(
        clean,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=clean[TARGET]
    )

    # --------------------------------------------------------
    # Step 7: Save train and test CSV files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    # --------------------------------------------------------
    # Step 8: Create a reproducibility manifest
    # --------------------------------------------------------

    manifest = {
        "source_file": str(SOURCE.relative_to(ROOT)),
        "train_file": str(train_path.relative_to(ROOT)),
        "test_file": str(test_path.relative_to(ROOT)),
        "rows_original": int(len(raw)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "target": TARGET,
        "feature_columns": FEATURE_COLUMNS,
        "excluded_post_contact_columns": POST_CONTACT_COLUMNS,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_sha256": file_sha256(train_path),
        "test_sha256": file_sha256(test_path),
    }

    manifest_path = OUTPUT_DIR / "split_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Step 9: Print audit and validation summary
    # --------------------------------------------------------

    print(
        f"\nInput: {len(raw):,} rows "
        f"x {len(raw.columns)} columns"
    )

    print(
        f"Cleaned: {len(clean):,} rows "
        f"x {len(clean.columns)} columns"
    )

    print(f"Predictive features: {len(FEATURE_COLUMNS)}")

    missing_cells = int(
        clean[FEATURE_COLUMNS].isna().sum().sum()
    )

    print(
        "Missing feature cells remaining "
        f"(for training-only imputation): {missing_cells}"
    )

    for name, part in [("Train", train), ("Test", test)]:

        counts = (
            part[TARGET]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"{name}: {len(part):,} rows; "
            f"target counts {counts}; "
            f"positive rate {part[TARGET].mean():.2%}"
        )

    print("\nOutput files:")
    print(train_path)
    print(test_path)
    print("Saved manifest:", manifest_path)

    print("\nDATA PREPARATION: SUCCESS")

    return train, test


if __name__ == "__main__":
    prepare_data()
