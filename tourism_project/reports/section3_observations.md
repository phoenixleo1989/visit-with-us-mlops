
## Section 3: Model Building and Experiment Tracking

### 1. Model Selection

A Random Forest Classifier was selected to predict
historical travel-package purchase propensity.

The model was implemented within a scikit-learn Pipeline
containing numeric and categorical preprocessing stages.

Numeric features were processed using median imputation.
Categorical features were processed using most-frequent
imputation and one-hot encoding.

All preprocessing estimators were fitted inside the
cross-validation process to prevent data leakage.

### 2. Hyperparameter Optimization

Eight Random Forest hyperparameter combinations were
evaluated using three-fold stratified cross-validation.

The primary optimization criterion was average precision
because the purchasing class represents a minority
of observations in the dataset.

**Best hyperparameters:** `{'model__max_depth': 12, 'model__min_samples_leaf': 1, 'model__n_estimators': 200}`

**Best cross-validation average precision:**
0.7146

### 3. Model Evaluation

The selected model was evaluated against the independent
20% holdout test dataset.

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 0.8680 |
| Balanced Accuracy | 0.7099 |
| Precision | 0.7660 |
| Recall | 0.4528 |
| F1-Score | 0.5692 |
| ROC-AUC | 0.9039 |
| Average Precision | 0.7125 |

The confusion matrix was generated to examine
correct and incorrect classifications.

The ROC curve and precision-recall curve were generated
to assess discrimination across prediction thresholds.

### 4. Business Insights

When customers were ranked by their predicted probability
of purchasing a travel package, the highest-ranked 20%
contained **66.04%**
of observed purchasers.

This represents a lift of
**3.29x**
relative to selecting customers without model-based ranking.

The result demonstrates potential business value for
customer prioritization and marketing resource allocation.

This lift is based on historical observed outcomes and
does not establish the causal effectiveness of outreach.

### 5. Experiment Tracking

MLflow was used to maintain a reproducible record
of hyperparameter experimentation.

Eight candidate configurations were logged,
including their parameters and validation metrics.

A separate best-model run recorded the selected model,
holdout evaluation metrics and related reports.

The MLflow tracking metadata was stored using SQLite,
while model and report artifacts were stored separately.

### 6. Model Serialization and Reproducibility

The complete fitted preprocessing and classification
pipeline was serialized as model.joblib.

A separate smoke test verified that the saved pipeline
could be loaded and generate valid classification
probabilities.

A GitHub Actions workflow was also configured to run
data registration, preparation, artifact verification,
model training and automatic model publication.

Successful remote execution will be verified after
the repository is published.

### 7. Limitations

The dataset's target variable represents historical
travel-package purchases in general.

It does not confirm purchases of the newly proposed
Wellness Tourism Package.

The resulting model should therefore be regarded as a
historical purchase-propensity proxy rather than a
validated wellness-specific predictor.

Before production use, further evaluation should address
probability calibration, performance drift, model fairness,
and the use of customer data in marketing activities.

### 8. Conclusion

An end-to-end local model-building pipeline was
implemented successfully, including preprocessing,
hyperparameter tuning, experiment tracking, model
evaluation and serialization.

The saved Random Forest pipeline is ready for
integration with the Streamlit prediction application.
