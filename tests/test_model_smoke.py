# ============================================================
# VISIT WITH US - SAVED MODEL SMOKE TEST
# ============================================================

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "tourism_project/deployment/model.joblib"
)

TEST_PATH = (
    ROOT / "tourism_project/outputs/test.csv"
)


def test_saved_model():

    # --------------------------------------------------------
    # Validate required input artifacts
    # --------------------------------------------------------

    assert MODEL_PATH.is_file(), "Model artifact missing."
    assert TEST_PATH.is_file(), "Test dataset missing."

    # --------------------------------------------------------
    # Load serialized preprocessing + model pipeline
    # --------------------------------------------------------

    model = joblib.load(MODEL_PATH)

    test_data = pd.read_csv(TEST_PATH)

    X_sample = test_data.drop(
        columns=["ProdTaken"]
    ).head(10)

    # --------------------------------------------------------
    # Generate predictions and probabilities
    # --------------------------------------------------------

    predictions = model.predict(X_sample)

    probabilities = model.predict_proba(X_sample)

    # --------------------------------------------------------
    # Validate model outputs
    # --------------------------------------------------------

    assert len(predictions) == 10

    assert probabilities.shape == (10, 2)

    assert set(predictions).issubset({0, 1})

    assert np.isfinite(probabilities).all()

    assert np.all(
        (probabilities >= 0)
        & (probabilities <= 1)
    )

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0
    )

    print("=" * 50)
    print("MODEL SMOKE TEST")
    print("=" * 50)

    print("Model loaded successfully.")
    print("Records tested:", len(X_sample))
    print("Predictions:", predictions.tolist())

    print(
        "Purchase probabilities:",
        probabilities[:, 1].round(4).tolist()
    )

    print("\nMODEL SMOKE TEST: PASSED")


if __name__ == "__main__":
    test_saved_model()
