
## Section 4 — Streamlit Application Development

### 1. Application Objective

An interactive Streamlit application was developed to
estimate customer purchase propensity using the
Random Forest model trained in Section 3.

The application is designed to support the identification
of prospective customers before sales interaction.

### 2. Model Integration

The application loads the saved `model.joblib` artifact
using joblib.

Streamlit resource caching is used to avoid repeated
model loading during application interaction.

The complete preprocessing and classification pipeline
is preserved within the serialized model.

### 3. Customer Input Interface

The user interface provides 13 inputs covering customer
demographics, occupation, designation, economic profile
and historical travel preferences.

Inputs are validated and converted into a one-row pandas
DataFrame containing the exact feature names required
by the trained model.

### 4. Prediction Outputs

The application generates a predicted purchase class
and an estimated positive-class probability.

It also displays an illustrative marketing-priority
category and the customer input DataFrame.

A CSV export allows the input record to be downloaded.

The model's held-out test ROC-AUC was
**0.9039**.

The model's held-out average precision was
**0.7125**.

### 5. Deployment Dependencies

A dedicated deployment requirements file was created,
containing Streamlit, scikit-learn, pandas, NumPy and
joblib.

Dependency versions were recorded from the model's
Colab runtime to improve reproducibility.

### 6. Testing and Verification

Three automated Streamlit application tests were
implemented to verify:

1. Successful application initialization.
2. Customer input capture and prediction generation.
3. Rejection of inconsistent traveller information.

The deployment readiness audit also verifies the model,
application source, dependencies and feature schema.

### 7. Deployment Architecture

The application entrypoint is:

`tourism_project/deployment/app.py`

The model artifact is:

`tourism_project/deployment/model.joblib`

The application is prepared for public deployment
through Streamlit Community Cloud after GitHub
repository publication and successful model commit.

**Deployment status at this stage: locally tested;
public deployment pending.**

### 8. Limitations

The model predicts historical travel-package
purchase propensity rather than validated
Wellness Tourism Package purchase propensity.

Predicted probabilities have not been independently
calibrated, and the displayed prioritization bands
are illustrative rather than validated thresholds.

### 9. Conclusion

A model-integrated Streamlit prediction application
has been developed with structured customer inputs,
probability-based outputs, input validation,
deployment dependencies and automated testing.

The next step is to publish the repository, run the
complete GitHub Actions pipeline, and deploy the app
on Streamlit Community Cloud.
