"""
Evaluation & Visualization Module

Generates rigorous metrics and publication-quality plots:
- Accuracy, Precision, Recall, F1-score, Confusion Matrix, Classification Report
- Model Comparison Bar Chart (Paper Fig. 3)
- 2D Scatter Plot with Linear Decision Boundaries (Paper Figs. 4 & 5)
- Class Distribution & Feature Histogram
- 19-Span Optical Attenuation & Gain Profiles
- Persists metrics to outputs/metrics/ and figures to outputs/figures/
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_models(
    models_dir: str = "outputs/models",
    figures_dir: str = "outputs/figures",
    metrics_dir: str = "outputs/metrics",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates saved models on the hold-out 25% test set (1,500 samples)
    and generates all required comparison plots and JSON metrics.
    """
    models_path = Path(models_dir)
    figs_path = Path(figures_dir)
    metrics_path = Path(metrics_dir)

    figs_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # 1. Load test data split
    split_data = np.load(models_path / "dataset_split.npz")
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    X_train_proc = split_data["X_train_proc"]
    X_test_proc = split_data["X_test_proc"]
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # 2. Load trained models
    lr_model = joblib.load(models_path / "logistic_regression.joblib")
    newton_lr_model = joblib.load(models_path / "newton_logistic_regression.joblib")
    gda_model = joblib.load(models_path / "gda.joblib")
    svm_model = joblib.load(models_path / "linear_svm.joblib")

    models = {
        "Logistic Regression (Newton-CG)": lr_model,
        "Logistic Regression (Pure Newton)": newton_lr_model,
        "Gaussian Discriminant Analysis (GDA)": gda_model,
        "Support Vector Machine (Linear SVM)": svm_model,
    }

    results = {}
    reports = {}
    cms = {}

    print("\n================== MODEL EVALUATION (TEST SET: 1500 SAMPLES) ==================")

    for name, model in models.items():
        y_pred = model.predict(X_test_proc)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Intercept and coefficients
        coef = getattr(model, "coef_", None)
        intercept = getattr(model, "intercept_", None)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Confusion_Matrix": cm,
            "Intercept": float(intercept[0]) if intercept is not None else None,
            "Coefficients": [float(c) for c in coef[0]] if coef is not None else None,
        }
        reports[name] = classification_report(y_test, y_pred, target_names=["Healthy (0)", "Failing (1)"])
        cms[name] = np.array(cm)

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if intercept is not None and coef is not None:
            print(f"Decision Boundary: {intercept[0]:.4f} + ({coef[0][0]:.4f})*X1 + ({coef[0][1]:.4f})*X2 = 0")

    # Save metrics JSON
    with open(metrics_path / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save comparison CSV
    summary_df = pd.DataFrame({
        name: {
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1-Score": m["F1-Score"],
        }
        for name, m in results.items()
    }).T
    summary_df.to_csv(metrics_path / "model_comparison.csv")
    print("\nSummary comparison table:\n", summary_df)

    # ==================== VISUALIZATION 1: MODEL COMPARISON (Fig 3) ====================
    plt.figure(figsize=(10, 6), dpi=300)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plot_df = summary_df.loc[[
        "Logistic Regression (Newton-CG)",
        "Gaussian Discriminant Analysis (GDA)",
        "Support Vector Machine (Linear SVM)",
    ]][metrics_to_plot].reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ax = sns.barplot(data=plot_df, x="index", y="Score", hue="Metric", palette=palette)
    plt.title("Classifier Performance Comparison (75% Train / 25% Test)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Classifier", fontsize=12, labelpad=8)
    plt.ylabel("Score", fontsize=12, labelpad=8)
    plt.ylim(0.85, 1.02)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 3),
                        textcoords="offset points")

    plt.tight_layout()
    plt.savefig(figs_path / "model_comparison.png")
    plt.close()

    # ==================== VISUALIZATION 2: DECISION BOUNDARIES (Figs 4 & 5) ====================
    plt.figure(figsize=(11, 8), dpi=300)
    # Scatter plot of test data
    mask0 = (y_test == 0)
    mask1 = (y_test == 1)

    plt.scatter(
        X_test[mask0, 0],
        X_test[mask0, 1],
        c="#1f77b4",
        marker="o",
        s=35,
        alpha=0.6,
        label="Healthy Links (OSNR Normal, y=0)",
        edgecolors="none",
    )
    plt.scatter(
        X_test[mask1, 0],
        X_test[mask1, 1],
        c="#d62728",
        marker="^",
        s=45,
        alpha=0.8,
        label="Failing / Degraded Links (y=1)",
        edgecolors="k",
        linewidths=0.4,
    )

    # Plot decision boundaries: theta0 + theta1*X1 + theta2*X2 = 0 => X2 = -(theta1*X1 + theta0)/theta2
    x1_min, x1_max = X_test[:, 0].min() - 2, X_test[:, 0].max() + 2
    x1_line = np.linspace(x1_min, x1_max, 300)

    # Boundary 1: Logistic Regression
    b_lr = lr_model.intercept_[0]
    w_lr = lr_model.coef_[0]
    if abs(w_lr[1]) > 1e-6:
        x2_lr = -(w_lr[0] * x1_line + b_lr) / w_lr[1]
        plt.plot(x1_line, x2_lr, color="#2ca02c", linestyle="-", linewidth=2.5,
                 label=f"Logistic Regression Boundary: {b_lr:.2f} + {w_lr[0]:.2f}X1 + {w_lr[1]:.2f}X2 = 0")

    # Boundary 2: GDA
    b_gda = gda_model.intercept_[0]
    w_gda = gda_model.coef_[0]
    if abs(w_gda[1]) > 1e-6:
        x2_gda = -(w_gda[0] * x1_line + b_gda) / w_gda[1]
        plt.plot(x1_line, x2_gda, color="#ff7f0e", linestyle="--", linewidth=2.5,
                 label=f"GDA Boundary: {b_gda:.2f} + {w_gda[0]:.2f}X1 + {w_gda[1]:.2f}X2 = 0")

    # Boundary 3: Linear SVM
    b_svm = svm_model.intercept_[0]
    w_svm = svm_model.coef_[0]
    if abs(w_svm[1]) > 1e-6:
        x2_svm = -(w_svm[0] * x1_line + b_svm) / w_svm[1]
        plt.plot(x1_line, x2_svm, color="#9467bd", linestyle=":", linewidth=2.5,
                 label=f"Linear SVM Boundary: {b_svm:.2f} + {w_svm[0]:.2f}X1 + {w_svm[1]:.2f}X2 = 0")

    plt.xlim(x1_min, x1_max)
    plt.ylim(X_test[:, 1].min() - 4, X_test[:, 1].max() + 4)
    plt.title("Optical Link Classifier Decision Boundaries (p(y|x) = 0.5)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel(r"$X_1 = \sum_{n=0}^{18} (\mathrm{Ampgain}_n - \mathrm{Targetgain}_n)$ [dB]", fontsize=12, labelpad=8)
    plt.ylabel(r"$X_2 = \sum_{n=0}^{18} (\mathrm{Target\_span\_loss}_n - \mathrm{span\_loss}_n)$ [dB]", fontsize=12, labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, facecolor="white", framealpha=0.95, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(figs_path / "decision_boundaries.png")
    plt.close()

    # ==================== VISUALIZATION 3: CONFUSION MATRICES ====================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    chosen_models = [
        ("Logistic Regression", cms["Logistic Regression (Newton-CG)"]),
        ("Gaussian Discriminant Analysis", cms["Gaussian Discriminant Analysis (GDA)"]),
        ("Linear SVM", cms["Support Vector Machine (Linear SVM)"]),
    ]

    for ax, (title, matrix) in zip(axes, chosen_models):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Healthy (0)", "Failing (1)"],
            yticklabels=["Healthy (0)", "Failing (1)"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=11, labelpad=6)
        ax.set_ylabel("True Class", fontsize=11, labelpad=6)

    plt.suptitle("Confusion Matrices on Test Set (1500 Samples)", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(figs_path / "confusion_matrices.png", bbox_inches="tight")
    plt.close()

    # ==================== VISUALIZATION 4: CLASS & FEATURE DISTRIBUTION ====================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    # Subplot 1: Class balance
    labels, counts = np.unique(y_test, return_counts=True)
    class_names = ["Healthy (0)", "Failing (1)"]
    axes[0].bar(class_names, counts, color=["#1f77b4", "#d62728"], width=0.5, edgecolor="black")
    axes[0].set_title("Test Set Class Distribution", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=11)
    for i, c in enumerate(counts):
        axes[0].text(i, c + 15, f"{c} ({c/len(y_test)*100:.1f}%)", ha="center", fontweight="bold")
    axes[0].set_ylim(0, max(counts) * 1.15)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Subplot 2: Feature X1 Histogram
    sns.kdeplot(X_test[mask0, 0], label="Healthy", ax=axes[1], color="#1f77b4", fill=True, alpha=0.3)
    sns.kdeplot(X_test[mask1, 0], label="Failing", ax=axes[1], color="#d62728", fill=True, alpha=0.3)
    axes[1].set_title(r"Distribution of Feature $X_1$ (Amp Diff)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(r"$X_1$ [dB]", fontsize=11)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Feature X2 Histogram
    sns.kdeplot(X_test[mask0, 1], label="Healthy", ax=axes[2], color="#1f77b4", fill=True, alpha=0.3)
    sns.kdeplot(X_test[mask1, 1], label="Failing", ax=axes[2], color="#d62728", fill=True, alpha=0.3)
    axes[2].set_title(r"Distribution of Feature $X_2$ (Loss Diff)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel(r"$X_2$ [dB]", fontsize=11)
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Dataset Characteristics & Feature Distributions", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(figs_path / "class_distribution.png", bbox_inches="tight")
    plt.close()

    # ==================== VISUALIZATION 5: 19-SPAN PROFILES ====================
    # Demonstrate span-level attenuation and gain profiles for a normal vs failing link
    raw_csv = Path("data/optical_network_raw.csv")
    if raw_csv.exists():
        raw_df = pd.read_csv(raw_csv)
        healthy_sample = raw_df[raw_df["label"] == 0].iloc[0]
        failing_sample = raw_df[raw_df["label"] == 1].iloc[0]

        spans = np.arange(19)
        h_loss = [healthy_sample[f"span_loss_span_{s}"] for s in spans]
        f_loss = [failing_sample[f"span_loss_span_{s}"] for s in spans]
        h_gain = [healthy_sample[f"amplifier_gain_span_{s}"] for s in spans]
        f_gain = [failing_sample[f"amplifier_gain_span_{s}"] for s in spans]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, dpi=300)

        ax1.plot(spans, h_loss, "o-", color="#1f77b4", label="Healthy Link Span Loss", linewidth=1.8)
        ax1.plot(spans, f_loss, "s--", color="#d62728", label="Failing Link Span Loss (Localized Spike)", linewidth=2.0)
        ax1.axhline(16.0, color="gray", linestyle=":", label="Nominal Target Loss (16.0 dB)")
        ax1.set_ylabel("Span Loss (dB)", fontsize=11)
        ax1.set_title("Span-by-Span Optical Telemetry Profile Across 19 Spans", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="upper left")

        ax2.plot(spans, h_gain, "o-", color="#2ca02c", label="Healthy Link Amp Gain", linewidth=1.8)
        ax2.plot(spans, f_gain, "^--", color="#ff7f0e", label="Failing Link Amp Gain (Gain Deficit)", linewidth=2.0)
        ax2.axhline(16.0, color="gray", linestyle=":", label="Nominal Target Gain (16.0 dB)")
        ax2.set_xlabel("Span Index (0 to 18)", fontsize=11)
        ax2.set_ylabel("Amplifier Gain (dB)", fontsize=11)
        ax2.set_xticks(spans)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(figs_path / "span_loss_profile.png")
        plt.close()

    print(f"\nAll evaluation plots saved to '{figures_dir}/'.")
    print(f"Metrics saved to '{metrics_dir}/model_metrics.json'.")

    return results


if __name__ == "__main__":
    evaluate_models()
