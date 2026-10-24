# ============================================================
# STREAMLIT APPLICATION AUTOMATED TESTS
# ============================================================

from pathlib import Path

import joblib
import pandas as pd

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]

APP_PATH = (
    ROOT / "tourism_project/deployment/app.py"
)

MODEL_PATH = (
    ROOT / "tourism_project/deployment/model.joblib"
)

EXPECTED_FEATURES = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation"
]


# ------------------------------------------------------------
# TEST 1: Application loads without execution errors
# ------------------------------------------------------------

def test_app_loads():

    assert APP_PATH.is_file()
    assert MODEL_PATH.is_file()

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=60
    )

    app.run()

    assert len(app.exception) == 0, (
        "Application generated an exception."
    )

    assert len(app.title) >= 1
    assert len(app.button) >= 1


# ------------------------------------------------------------
# TEST 2: Input form generates valid predictions
# ------------------------------------------------------------

def test_app_generates_prediction():

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=60
    )

    app.run()

    assert len(app.exception) == 0

    # Submit the form using its default customer profile.
    app.button[0].click().run()

    assert len(app.exception) == 0

    assert "last_input_df" in app.session_state
    assert "last_probability" in app.session_state
    assert "last_prediction" in app.session_state

    input_df = app.session_state["last_input_df"]

    assert isinstance(input_df, pd.DataFrame)

    assert input_df.shape == (1, 13)

    assert list(input_df.columns) == EXPECTED_FEATURES

    probability = app.session_state["last_probability"]

    assert 0 <= probability <= 1

    prediction = app.session_state["last_prediction"]

    assert prediction in [0, 1]

    # Independently verify the saved model's probability.
    trained_model = joblib.load(MODEL_PATH)

    positive_index = list(
        trained_model.classes_
    ).index(1)

    direct_probability = float(
        trained_model.predict_proba(input_df)[0][
            positive_index
        ]
    )

    assert abs(probability - direct_probability) < 1e-10


# ------------------------------------------------------------
# TEST 3: Reject logically inconsistent traveller inputs
# ------------------------------------------------------------

def test_invalid_traveller_configuration():

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=60
    )

    app.run()

    # Invalid case: 3 children with only 2 visitors.
    app.number_input(key="travel_party").set_value(2)

    app.number_input(
        key="children_under_5"
    ).set_value(3)

    app.button[0].click().run()

    assert len(app.exception) == 0

    assert len(app.error) >= 1

    assert "last_probability" not in app.session_state
