# ============================================================
# VISIT WITH US - MLOPS PROJECT
# SECTION 3: MODEL BUILDING AND EXPERIMENT TRACKING
# ============================================================

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlflow.models import infer_signature

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
    PrecisionRecallDisplay
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from schema import (
    TARGET,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "tourism_project/outputs"
MODEL_DIR = ROOT / "tourism_project/deployment"
REPORT_DIR = ROOT / "tourism_project/reports"
TRACKING_DIR = ROOT / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
TRACKING_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"
MODEL_PATH = MODEL_DIR / "model.joblib"

RANDOM_STATE = 42

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

mlflow.set_experiment(
    "VisitWithUs_Purchase_Propensity"
)


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

def load_data():

    if not TRAIN_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset missing: {TRAIN_PATH}"
        )

    if not TEST_PATH.is_file():
        raise FileNotFoundError(
            f"Testing dataset missing: {TEST_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    expected = set(FEATURE_COLUMNS + [TARGET])

    if set(train.columns) != expected:
        raise ValueError("Training schema mismatch.")

    if set(test.columns) != expected:
        raise ValueError("Testing schema mismatch.")

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train[TARGET].astype(int)

    X_test = test[FEATURE_COLUMNS].copy()
    y_test = test[TARGET].astype(int)

    print("\nTraining observations:", len(X_train))
    print("Testing observations:", len(X_test))
    print("Predictive features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test


# ============================================================
# 3. DEFINE PREPROCESSING PIPELINES
# ============================================================

def create_pipeline():

    # Numeric features: impute using the training-fold median.
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    # Categorical features:
    # Impute missing categories and one-hot encode them.
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_COLUMNS
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_COLUMNS
            )
        ],
        remainder="drop"
    )

    # Balanced class weighting addresses minority-class
    # underrepresentation without oversampling test data.
    classifier = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=1
    )

    # Complete reusable prediction pipeline.
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier)
        ]
    )

    return pipeline


# ============================================================
# 4. DEFINE HYPERPARAMETER SEARCH
# ============================================================

def tune_model(pipeline, X_train, y_train):

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [6, 12],
        "model__min_samples_leaf": [1, 3]
    }

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=1,
        error_score="raise"
    )

    print("\nStarting hyperparameter tuning...")

    search.fit(X_train, y_train)

    print("\nBEST HYPERPARAMETERS")
    print(search.best_params_)

    print(
        "Best cross-validation average precision:",
        round(search.best_score_, 4)
    )

    return search


# ============================================================
# 5. LOG EVERY EXPERIMENT TO MLFLOW
# ============================================================

def log_search_experiments(search):

    results = search.cv_results_

    # Log each evaluated hyperparameter combination.
    for index, params in enumerate(results["params"]):

        with mlflow.start_run(
            run_name=f"candidate_{index + 1:02d}"
        ):

            mlflow.set_tag(
                "run_type",
                "hyperparameter_candidate"
            )

            mlflow.log_params({
                key: str(value)
                for key, value in params.items()
            })

            mlflow.log_metric(
                "cv_mean_average_precision",
                float(results["mean_test_score"][index])
            )

            mlflow.log_metric(
                "cv_std_average_precision",
                float(results["std_test_score"][index])
            )

            mlflow.log_metric(
                "cv_mean_train_average_precision",
                float(results["mean_train_score"][index])
            )

            mlflow.log_metric(
                "cv_rank",
                int(results["rank_test_score"][index])
            )

    print(
        "\nLogged",
        len(results["params"]),
        "hyperparameter experiments to MLflow."
    )


# ============================================================
# 6. EVALUATE THE BEST MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(
                y_test,
                probabilities
            )
        )
    }

    # Marketing-oriented metric:
    # How many positives appear in the highest-ranked 20%?
    top_n = int(np.ceil(0.20 * len(y_test)))

    ranking = np.argsort(-probabilities)

    captured = int(
        y_test.to_numpy()[ranking[:top_n]].sum()
    )

    total_positives = int(y_test.sum())

    capture_rate = captured / total_positives

    top_positive_rate = captured / top_n

    overall_positive_rate = float(y_test.mean())

    lift = top_positive_rate / overall_positive_rate

    metrics["top_20_capture_rate"] = float(capture_rate)
    metrics["top_20_lift"] = float(lift)

    print("\nTEST SET PERFORMANCE")
    print("-" * 45)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["No Purchase", "Purchase"],
        cmap="Blues"
    )

    plt.title("Random Forest - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # Save ROC curve
    # --------------------------------------------------------

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title("Random Forest - ROC Curve")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "roc_curve.png",
        dpi=160
    )

    plt.close()

    # --------------------------------------------------------
    # Save Precision-Recall curve
    # --------------------------------------------------------

    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities
    )

    plt.title("Random Forest - Precision Recall")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "precision_recall_curve.png",
        dpi=160
    )

    plt.close()

    return metrics


# ============================================================
# 7. SAVE MODEL, METRICS AND FEATURE IMPORTANCES
# ============================================================

def save_outputs(search, metrics):

    best_model = search.best_estimator_

    # Save the complete preprocessing + classifier pipeline.
    joblib.dump(
        best_model,
        MODEL_PATH,
        compress=3
    )

    results = pd.DataFrame(search.cv_results_)

    results.to_csv(
        REPORT_DIR / "cv_results.csv",
        index=False
    )

    # Export original and encoded feature importance.
    feature_names = (
        best_model
        .named_steps["preprocess"]
        .get_feature_names_out()
    )

    importances = (
        best_model
        .named_steps["model"]
        .feature_importances_
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    )

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False
    )

    # JSON-safe hyperparameter values.
    best_params = {
        key: int(value)
        for key, value in search.best_params_.items()
    }

    report = {
        "algorithm": "RandomForestClassifier",
        "selection_metric": "average_precision",
        "cross_validation_folds": 3,
        "best_cv_average_precision": float(
            search.best_score_
        ),
        "best_parameters": best_params,
        "test_metrics": metrics
    }

    metrics_path = REPORT_DIR / "model_metrics.json"

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nMODEL ARTIFACT SAVED:")
    print(MODEL_PATH)

    print("\nMODEL REPORT SAVED:")
    print(metrics_path)

    return best_model, report


# ============================================================
# 8. MLFLOW BEST-MODEL TRACKING
# ============================================================

def log_best_model(
    search,
    model,
    report,
    X_train
):

    with mlflow.start_run(
        run_name="BEST_RANDOM_FOREST_MODEL"
    ) as run:

        mlflow.set_tag(
            "run_type",
            "selected_best_model"
        )

        mlflow.set_tag(
            "model_purpose",
            "pre_contact_purchase_propensity"
        )

        mlflow.log_params({
            key: str(value)
            for key, value in search.best_params_.items()
        })

        mlflow.log_param(
            "cv_folds", 3
        )

        mlflow.log_param(
            "selection_metric",
            "average_precision"
        )

        mlflow.log_param(
            "class_weight",
            "balanced"
        )

        mlflow.log_metrics({
            "best_cv_average_precision":
                report["best_cv_average_precision"],
            **report["test_metrics"]
        })

        # Log evaluation reports and charts.
        mlflow.log_artifacts(
            str(REPORT_DIR),
            artifact_path="evaluation_reports"
        )

        # Record the model's expected input schema.
        # No raw customer examples are logged.
        signature = infer_signature(
            X_train.head(3),
            model.predict(X_train.head(3))
        )

        # Log a versioned MLflow model in addition
        # to the joblib deployment artifact.
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="purchase_propensity_pipeline",
            signature=signature,
            serialization_format="cloudpickle"
        )

        print("\nMLflow Run ID:", run.info.run_id)
        print("MLflow Model URI:", model_info.model_uri)


# ============================================================
# 9. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("VISIT WITH US | MODEL TRAINING PIPELINE")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()

    pipeline = create_pipeline()

    search = tune_model(
        pipeline,
        X_train,
        y_train
    )

    log_search_experiments(search)

    best_model = search.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    best_model, report = save_outputs(
        search,
        metrics
    )

    log_best_model(
        search,
        best_model,
        report,
        X_train
    )

    print("\n" + "=" * 60)
    print("MODEL BUILDING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
