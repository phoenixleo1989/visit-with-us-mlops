# ============================================================
# VISIT WITH US - CUSTOMER PURCHASE PROPENSITY APPLICATION
# ============================================================
#
# Purpose:
# Score prospective customers before sales contact using
# the trained Random Forest classification pipeline.
#
# Model source:
# tourism_project/deployment/model.joblib
#
# The prediction is a historical package-purchase propensity
# proxy. It is not validated specifically for wellness packages.
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Visit With Us | Purchase Propensity",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).resolve().with_name(
    "model.joblib"
)

FEATURE_COLUMNS = [
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


# ============================================================
# 2. APPLICATION STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    .hero-card {
        background: linear-gradient(
            120deg,
            #102A43 0%,
            #183D5D 55%,
            #146B75 100%
        );
        padding: 32px;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 24px;
    }

    .hero-card h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2.1rem;
    }

    .hero-card p {
        color: #DBEAFE;
        margin-top: 12px;
        font-size: 1rem;
    }

    .eyebrow {
        color: #93C5FD;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.13em;
        margin-bottom: 10px;
    }

    .result-label {
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 8px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_file):
    """
    Load the fitted preprocessing + Random Forest pipeline.

    Streamlit caches the model so it is not loaded again
    during every user interaction.
    """
    return joblib.load(model_file)


if not MODEL_PATH.is_file():
    st.error(
        "The trained model was not found. "
        "GitHub Actions must publish model.joblib "
        "before this application can be deployed."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "The trained model could not be loaded. "
        "Verify the scikit-learn and joblib versions."
    )
    st.exception(exc)
    st.stop()


# Verify that the UI matches the model's training schema.
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURE_COLUMNS:
        st.error(
            "The model input schema does not match "
            "the application feature configuration."
        )
        st.stop()


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("✈️ Visit With Us")

    st.caption("Predictive Marketing Intelligence")

    st.divider()

    st.subheader("About this application")

    st.write(
        "This application estimates the historical "
        "travel-package purchase propensity of "
        "a prospective customer."
    )

    st.info(
        "Predictions are based on a Random Forest model "
        "trained on historical tourism data."
    )

    st.subheader("Model information")

    st.write("**Algorithm:** Random Forest")
    st.write("**Input attributes:** 13")
    st.write("**Model output:** Binary class + probability")
    st.write("**Scoring stage:** Before sales contact")

    st.divider()

    st.caption(
        "Academic MLOps project | "
        "Visit With Us | Streamlit"
    )


# ============================================================
# 5. HEADER
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="eyebrow">CUSTOMER INTELLIGENCE PLATFORM</div>
        <h1>Customer Purchase Propensity</h1>
        <p>
            Identify prospective customers using a trained
            machine-learning model and support more
            targeted travel marketing decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write(
    "Enter a prospective customer's profile below. "
    "The model uses customer attributes and travel "
    "preferences available before the sales pitch."
)


# ============================================================
# 6. CUSTOMER INPUT FORM
# ============================================================

with st.form("customer_prediction_form"):

    st.subheader("Customer Information")

    st.caption(
        "Complete the 13 attributes used during "
        "model training."
    )

    col1, col2, col3 = st.columns(3, gap="large")

    # --------------------------------------------------------
    # Column 1: Demographic profile
    # --------------------------------------------------------

    with col1:

        st.markdown("#### Personal profile")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=61,
            value=36,
            step=1,
            key="age"
        )

        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            key="gender"
        )

        marital_status = st.selectbox(
            "Marital status",
            options=[
                "Married",
                "Single",
                "Unmarried",
                "Divorced"
            ],
            key="marital_status"
        )

        occupation = st.selectbox(
            "Occupation",
            options=[
                "Salaried",
                "Small Business",
                "Large Business",
                "Freelancer"
            ],
            key="occupation"
        )

    # --------------------------------------------------------
    # Column 2: Economic and residential profile
    # --------------------------------------------------------

    with col2:

        st.markdown("#### Economic profile")

        city_tier = st.selectbox(
            "City tier",
            options=[1, 2, 3],
            format_func=lambda value: f"Tier {value}",
            key="city_tier"
        )

        designation = st.selectbox(
            "Designation",
            options=[
                "Executive",
                "Manager",
                "Senior Manager",
                "AVP",
                "VP"
            ],
            index=1,
            key="designation"
        )

        monthly_income = st.number_input(
            "Gross monthly income (₹)",
            min_value=1000,
            max_value=98678,
            value=22500,
            step=500,
            key="monthly_income"
        )

        own_car = st.selectbox(
            "Owns a car?",
            options=["No", "Yes"],
            key="own_car"
        )

        passport = st.selectbox(
            "Has a valid passport?",
            options=["No", "Yes"],
            key="passport"
        )

    # --------------------------------------------------------
    # Column 3: Travel preferences
    # --------------------------------------------------------

    with col3:

        st.markdown("#### Travel preferences")

        number_of_travellers = st.number_input(
            "Number of people visiting",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            key="travel_party"
        )

        number_of_children = st.number_input(
            "Children below age 5 visiting",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
            key="children_under_5"
        )

        preferred_star = st.selectbox(
            "Preferred property rating",
            options=[3, 4, 5],
            format_func=lambda value: f"{value} stars",
            index=1,
            key="property_rating"
        )

        number_of_trips = st.number_input(
            "Average annual trips",
            min_value=1,
            max_value=22,
            value=3,
            step=1,
            key="annual_trips"
        )

    st.divider()

    submitted = st.form_submit_button(
        "Calculate Purchase Propensity",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 7. CREATE DATAFRAME AND PREDICT
# ============================================================

if submitted:

    # Basic consistency check.
    if number_of_children >= number_of_travellers:

        st.error(
            "The number of children below age 5 must be "
            "smaller than the total number of people visiting."
        )

        st.session_state.pop("last_input_df", None)
        st.session_state.pop("last_probability", None)
        st.session_state.pop("last_prediction", None)

    else:

        # Construct the record using the exact feature names
        # and expected data types from model training.
        customer_data = {
            "Age": int(age),
            "CityTier": int(city_tier),
            "NumberOfPersonVisiting": int(
                number_of_travellers
            ),
            "PreferredPropertyStar": int(preferred_star),
            "NumberOfTrips": int(number_of_trips),
            "Passport": int(passport == "Yes"),
            "OwnCar": int(own_car == "Yes"),
            "NumberOfChildrenVisiting": int(
                number_of_children
            ),
            "MonthlyIncome": int(monthly_income),
            "Occupation": occupation,
            "Gender": gender,
            "MaritalStatus": marital_status,
            "Designation": designation
        }

        # Convert customer inputs into a one-row DataFrame.
        input_df = pd.DataFrame(
            [customer_data],
            columns=FEATURE_COLUMNS
        )

        try:

            # Predict binary outcome.
            prediction = int(
                model.predict(input_df)[0]
            )

            # Identify probability of the positive class.
            positive_index = list(
                model.classes_
            ).index(1)

            probability = float(
                model.predict_proba(input_df)[0][
                    positive_index
                ]
            )

            # Store the resulting DataFrame and predictions
            # in Streamlit session state.
            st.session_state["last_input_df"] = input_df
            st.session_state["last_prediction"] = prediction
            st.session_state["last_probability"] = probability

        except Exception as exc:

            st.error(
                "The customer could not be scored. "
                "Please check the inputs and model."
            )

            st.exception(exc)


# ============================================================
# 8. DISPLAY PREDICTION RESULTS
# ============================================================

if "last_probability" in st.session_state:

    probability = st.session_state["last_probability"]
    prediction = st.session_state["last_prediction"]
    input_df = st.session_state["last_input_df"]

    st.divider()

    st.header("Prediction Results")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Estimated Purchase Probability",
            value=f"{probability:.1%}"
        )

    with result_col2:

        predicted_label = (
            "Likely to purchase"
            if prediction == 1
            else "Not predicted to purchase"
        )

        st.metric(
            label="Predicted Classification",
            value=predicted_label
        )

    st.write("Estimated probability indicator")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    # Heuristic prioritization bands, not calibrated
    # decision thresholds.
    if probability >= 0.65:
        priority = "High"
    elif probability >= 0.35:
        priority = "Moderate"
    else:
        priority = "Lower"

    st.info(
        f"Illustrative marketing priority: **{priority}**. "
        "These priority bands are illustrative and "
        "have not been validated as business thresholds."
    )

    st.caption(
        "The classification comes from the trained model. "
        "The probability is a model estimate and has "
        "not been separately calibrated."
    )

    st.divider()

    st.subheader("Input Record Used for Prediction")

    # Display precisely the DataFrame submitted to the model.
    st.dataframe(
        input_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download Input Record (CSV)",
        data=input_df.to_csv(index=False).encode("utf-8"),
        file_name="customer_prediction_input.csv",
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 9. BUSINESS DISCLAIMER AND FOOTER
# ============================================================

st.divider()

st.warning(
    "**Important:** The training dataset records general "
    "historical travel-package purchases. It does not "
    "identify confirmed purchases of the new Wellness "
    "Tourism Package. This prediction is therefore a "
    "historical purchase-propensity proxy."
)

st.caption(
    "This application is an academic demonstration. "
    "It should not be used as the sole basis for "
    "marketing or consequential customer decisions."
)

st.caption(
    "Visit With Us | Automated ML Pipeline | "
    "Random Forest | MLflow | GitHub Actions | Streamlit"
)
