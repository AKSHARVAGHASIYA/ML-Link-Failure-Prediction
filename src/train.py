"""
Training Pipeline Module

Executes reproducible 75% train / 25% test split (matching paper's 4,500 train / 1,500 test).
Fits preprocessing strictly on training data (no leakage).
Trains:
  1. Logistic Regression (Newton-CG solver & Custom Newton's Method)
  2. Gaussian Discriminant Analysis (GDA)
  3. Support Vector Machine (Linear SVM)
Saves models and artifacts to outputs/models/.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_or_generate_dataset
from src.preprocessing import TelemetryPreprocessor, clean_raw_telemetry
from src.feature_engineering import extract_features_from_raw
from src.models import (
    NewtonLogisticRegression,
    build_logistic_regression,
    GaussianDiscriminantAnalysis,
    build_linear_svm,
)


def run_training_pipeline(
    data_dir: str = "data",
    output_dir: str = "outputs/models",
    random_state: int = 42,
    scale_features: bool = False,
) -> Dict[str, any]:
    """
    Executes end-to-end training pipeline.

    Note on scaling:
    In the original CS229 paper, features X1 and X2 are preserved in their native
    physical units (decibels, dB) so that the decision boundary coefficients directly
    reflect optical attenuation and gain margins (as reported in Table 1).
    We support both native dB scale and normalized feature representations.
    """
    models_path = Path(output_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Load or synthesize dataset (6,000 observations across 19 spans)
    raw_df, features_df = load_or_generate_dataset(data_dir=data_dir, random_state=random_state)

    # 2. Extract features X1 and X2, and label y
    X = features_df[["X1", "X2"]].values
    y = features_df["label"].values

    # 3. Reproducible 75% / 25% train/test split (4500 train / 1500 test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    # 4. Leakage-safe Preprocessor
    preprocessor = TelemetryPreprocessor(scale_features=scale_features)
    preprocessor.fit(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5. Model Initialization
    # Baseline: Scikit-learn Logistic Regression (Newton-CG solver)
    lr_model = build_logistic_regression(solver="newton-cg", C=1.0)
    # Reference: Custom Newton's Method Logistic Regression (CS229 exact derivation)
    newton_lr_model = NewtonLogisticRegression(max_iter=50, tol=1e-6)
    # GDA: Generative Gaussian Discriminant Analysis
    gda_model = GaussianDiscriminantAnalysis(reg_covar=1e-4)
    # Linear SVM: Support Vector Machine with linear kernel
    svm_model = build_linear_svm(C=1.0, random_state=random_state)

    # 6. Model Training
    print("Training Logistic Regression (Newton-CG)...")
    lr_model.fit(X_train_proc, y_train)

    print("Training Custom Newton's Method Logistic Regression...")
    newton_lr_model.fit(X_train_proc, y_train)

    print("Training Gaussian Discriminant Analysis (GDA)...")
    gda_model.fit(X_train_proc, y_train)

    print("Training Linear Support Vector Machine (Linear SVM)...")
    svm_model.fit(X_train_proc, y_train)

    # 7. Persist Models and Preprocessor
    joblib.dump(lr_model, models_path / "logistic_regression.joblib")
    joblib.dump(newton_lr_model, models_path / "newton_logistic_regression.joblib")
    joblib.dump(gda_model, models_path / "gda.joblib")
    joblib.dump(svm_model, models_path / "linear_svm.joblib")
    joblib.dump(preprocessor, models_path / "preprocessor.joblib")

    # Save split data for evaluation
    np.savez_compressed(
        models_path / "dataset_split.npz",
        X_train=X_train,
        X_test=X_test,
        X_train_proc=X_train_proc,
        X_test_proc=X_test_proc,
        y_train=y_train,
        y_test=y_test,
    )

    print(f"Training completed successfully. Models and split artifacts saved to '{output_dir}'.")

    return {
        "lr_model": lr_model,
        "newton_lr_model": newton_lr_model,
        "gda_model": gda_model,
        "svm_model": svm_model,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_proc": X_train_proc,
        "X_test_proc": X_test_proc,
    }


if __name__ == "__main__":
    run_training_pipeline()
