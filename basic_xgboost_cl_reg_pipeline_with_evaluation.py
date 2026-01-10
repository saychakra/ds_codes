from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor


def run_xgboost_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    categorical_cols: Optional[list] = None,
    numeric_cols: Optional[list] = None,
    cv: int = 5,
    model_params: Optional[dict] = None,
):
    if categorical_cols is None:
        categorical_cols = list()
    assert task_type in [], "Invalid task_type"

    # Separate features/target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Auto-detect column types if not passed
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
    if numeric_cols is None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocessing pipeline
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="mean"))])

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, numeric_cols), ("cat", cat_pipeline, categorical_cols)]
    )

    # Choose model
    if task_type == "classification":
        model = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", **(model_params or {})
        )
    else:
        model = XGBRegressor(**(model_params or {}))

    # Full pipeline
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])

    # Scoring metrics
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }
    else:
        scoring = {
            "mse": make_scorer(mean_squared_error),
            "mae": make_scorer(mean_absolute_error),
            "r2": "r2",
        }

    # Cross-validate
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)

    print(f"\n--- {task_type.upper()} CROSS-VALIDATION METRICS (mean over {cv} folds) ---")
    for metric, values in scores.items():
        if metric.startswith("test_"):
            print(f"{metric.replace('test_', '').upper()}: {np.mean(values):.4f}")

    # Fit on full data (optional)
    pipeline.fit(X, y)

    # Feature importance (from model stage)
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    return pipeline, feature_importance_df


## sample usage
df = pd.read_csv("your_data.csv")

pipeline, feat_imp = run_xgboost_pipeline(
    df=df,
    target_col="target",
    task_type="classification",  # or 'regression'
    cv=5,
)

## print the top 10 important features
print(feat_imp.head(10))
