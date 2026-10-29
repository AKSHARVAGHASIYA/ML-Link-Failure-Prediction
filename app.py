"""
Optical Network Link Failure Prediction & Localization Dashboard
Streamlit Application

Designed for Faculty Review & Live Demonstration (UE24CS352A)
Based on: "Link Failure Prediction and Localization in Cloud-Scale Networks Using Supervised Learning"
Author Reference: Zahra Bakhtiari (Stanford University CS229)
"""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import localize_fault, aggregate_features
from src.models.gda import GaussianDiscriminantAnalysis
from src.models.logistic_regression import NewtonLogisticRegression


# ==================== PAGE CONFIG & STYLES ====================
st.set_page_config(
    page_title="Optical Link Failure AI | Cloud Networks",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-healthy {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .status-failing {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.92rem;
        color: #92400E;
        margin: 14px 0;
    }
</style>
""", unsafe_allow_html=True)


# ==================== CACHED ARTIFACT LOADERS ====================
@st.cache_resource
def load_models_and_metrics():
    models_dir = Path("outputs/models")
    metrics_dir = Path("outputs/metrics")

    lr_model = joblib.load(models_dir / "logistic_regression.joblib")
    newton_lr = joblib.load(models_dir / "newton_logistic_regression.joblib")
    gda_model = joblib.load(models_dir / "gda.joblib")
    svm_model = joblib.load(models_dir / "linear_svm.joblib")
    preprocessor = joblib.load(models_dir / "preprocessor.joblib")

    with open(metrics_dir / "model_metrics.json", "r") as f:
        metrics = json.load(f)

    split_data = np.load(models_dir / "dataset_split.npz")

    return {
        "lr": lr_model,
        "newton_lr": newton_lr,
        "gda": gda_model,
        "svm": svm_model,
        "preprocessor": preprocessor,
        "metrics": metrics,
        "split": split_data,
    }


artifacts = load_models_and_metrics()


# ==================== SIDEBAR ====================
st.sidebar.image("https://img.icons8.com/color/96/server.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "🚀 Live Diagnosis & Localization",
        "📊 Model Benchmarks & Comparison",
        "🗺️ 19-Span Optical Telemetry",
        "📂 Dataset Exploration",
        "📚 Methodology & Architecture",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details:**")
st.sidebar.caption("• Course: UE24CS352A (ML)")
st.sidebar.caption("• Architecture: 19 Optical Spans")
st.sidebar.caption("• Telemetry: 2-min Bin Time Series")
st.sidebar.caption("• Split: 75% Train (4500) / 25% Test (1500)")


# ==================== PAGE 1: LIVE INFERENCE ====================
if page == "🚀 Live Diagnosis & Localization":
    st.markdown('<div class="main-header">Optical Link Health Diagnosis & Localization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Proactive supervised failure prediction and span-level root-cause localization for cloud backbones.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ OPERATIONAL DISTINCTION:</strong> Machine learning inferences reflect probabilistic telemetry patterns 
        (<em>P(failure | X)</em>). A predicted failure flags the lightpath for automated proactive rerouting and technician dispatch, 
        but does <strong>not</strong> substitute for physical OTDR (Optical Time-Domain Reflectometry) confirmation of a fiber cut.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.subheader("1. Inference Configuration")
        model_choice = st.selectbox(
            "Select Supervised Classifier",
            [
                "Logistic Regression (Newton-CG)",
                "Gaussian Discriminant Analysis (GDA)",
                "Support Vector Machine (Linear SVM)",
                "Logistic Regression (Pure Newton)",
            ],
        )

        model_map = {
            "Logistic Regression (Newton-CG)": artifacts["lr"],
            "Gaussian Discriminant Analysis (GDA)": artifacts["gda"],
            "Support Vector Machine (Linear SVM)": artifacts["svm"],
            "Logistic Regression (Pure Newton)": artifacts["newton_lr"],
        }
        selected_model = model_map[model_choice]

        st.markdown("#### Input Mode")
        input_mode = st.radio(
            "Select telemetry granularity:",
            ["Aggregated Link Features (X1, X2)", "19-Span Granular Telemetry (Full Diagnostics)"],
        )

        if input_mode == "Aggregated Link Features (X1, X2)":
            st.info("Feature definitions per paper Eq. (1)-(2):\n- X1: Sum of amplifier gain differences (dB)\n- X2: Sum of span loss differences (dB)")

            x1_val = st.slider(
                "X1: Total Amplifier Gain Diff (dB)",
                min_value=-25.0,
                max_value=10.0,
                value=0.2,
                step=0.1,
                help="Sum across 19 spans of (Ampgain - Targetgain). Negative values denote severe EDFA gain deficit/saturation.",
            )
            x2_val = st.slider(
                "X2: Total Span Loss Diff (dB)",
                min_value=-35.0,
                max_value=10.0,
                value=-0.1,
                step=0.1,
                help="Sum across 19 spans of (Target_span_loss - span_loss). Highly negative values denote excess physical attenuation.",
            )
            features = np.array([[x1_val, x2_val]])
            span_telemetry_available = False

        else:
            preset = st.selectbox(
                "Choose Operational Preset Scenario:",
                [
                    "Nominal Healthy Link (All 19 Spans Normal)",
                    "Span 7 High Attenuation (Macro-bend / Impending Cut)",
                    "Span 12 EDFA Pump Failure (Gain Deficit)",
                    "Span 3 Moderate Degradation (Dirty Connector)",
                ],
            )

            # Generate synthetic 19 spans based on preset
            nominal_loss = 16.0
            nominal_gain = 16.0
            span_losses = [nominal_loss + np.random.normal(0, 0.2) for _ in range(19)]
            target_span_losses = [nominal_loss] * 19
            target_gains = [nominal_gain] * 19
            amp_gains = [nominal_gain + np.random.normal(0, 0.2) for _ in range(19)]

            if "Span 7 High Attenuation" in preset:
                span_losses[7] += 7.5  # 7.5 dB excess loss
                amp_gains[7] -= 2.0
            elif "Span 12 EDFA Pump Failure" in preset:
                amp_gains[12] -= 6.8  # 6.8 dB gain drop
                span_losses[12] += 0.8
            elif "Span 3 Moderate Degradation" in preset:
                span_losses[3] += 3.8
                amp_gains[3] -= 1.5

            # Compute X1 and X2
            x1_spans = np.array(amp_gains) - np.array(target_gains)
            x2_spans = np.array(target_span_losses) - np.array(span_losses)
            X1_calc, X2_calc = aggregate_features(x1_spans.reshape(1, -1), x2_spans.reshape(1, -1))
            features = np.array([[X1_calc[0], X2_calc[0]]])
            span_telemetry_available = True

            st.write(f"**Computed Aggregated Features:** $X_1 = {X1_calc[0]:.2f}\\text{{ dB}}$, $X_2 = {X2_calc[0]:.2f}\\text{{ dB}}$")

    with col2:
        st.subheader("2. AI Diagnosis Results")

        # Run inference
        feat_proc = artifacts["preprocessor"].transform(features)
        prediction = selected_model.predict(feat_proc)[0]

        # Get probabilities if supported
        if hasattr(selected_model, "predict_proba"):
            probs = selected_model.predict_proba(feat_proc)[0]
            fail_prob = float(probs[1])
            healthy_prob = float(probs[0])
        else:
            fail_prob = 1.0 if prediction == 1 else 0.0
            healthy_prob = 1.0 - fail_prob

        # Display result card
        if prediction == 0:
            st.markdown(
                f'<div class="status-healthy">✅ PREDICTED CONDITION: HEALTHY LINK<br>'
                f'<small>Confidence: {healthy_prob*100:.1f}% | Failure Probability: {fail_prob*100:.1f}%</small></div>',
                unsafe_allow_html=True,
            )
            st.success("The end-to-end optical link meets all baseline OSNR requirements. Transmission is stable.")
        else:
            st.markdown(
                f'<div class="status-failing">🚨 PREDICTED CONDITION: MARGINAL / FAILING LINK<br>'
                f'<small>Confidence: {fail_prob*100:.1f}% | Risk Level: Critical</small></div>',
                unsafe_allow_html=True,
            )
            st.error("Proactive Action Recommended: Reroute cloud traffic away from this fiber route before hard customer outage.")

        st.markdown("#### Fault Localization Status")
        if span_telemetry_available:
            loc_result = localize_fault(span_losses, target_span_losses, amp_gains, target_gains)
            if loc_result["failing_span"] >= 0:
                st.warning(f"**Culprit Span Identified:** Span #{loc_result['failing_span']}")
                st.write(f"• **Anomaly Type:** {loc_result['anomaly_metric']}")
                st.write(f"• **Maximum Telemetry Deviation:** {loc_result['max_deviation_db']} dB")
                st.write(f"• **Diagnosis:** {loc_result['diagnosis']}")
            else:
                st.info("No single span exhibits anomalous degradation above thermal noise threshold.")
        else:
            st.warning(
                "**⚠️ Localization Unavailable:** Aggregated features $(X_1, X_2)$ identify whether the link is failing overall, "
                "but cannot localize which of the 19 spans contains the anomaly without per-span telemetry. "
                "Switch input mode to '19-Span Granular Telemetry' above to test localization."
            )

        # 2D Decision Space Visualization
        st.markdown("#### Position in Decision Space")
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        X_test = artifacts["split"]["X_test"]
        y_test = artifacts["split"]["y_test"]

        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#1f77b4", s=10, alpha=0.3, label="Healthy (Test)")
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#d62728", s=12, alpha=0.4, label="Failing (Test)")

        # Decision boundary
        coef = getattr(selected_model, "coef_", None)
        intercept = getattr(selected_model, "intercept_", None)
        if coef is not None and intercept is not None and abs(coef[0][1]) > 1e-6:
            x_range = np.linspace(-25, 10, 100)
            y_bound = -(coef[0][0] * x_range + intercept[0]) / coef[0][1]
            ax.plot(x_range, y_bound, "k--", linewidth=1.8, label="Decision Boundary")

        # Current point
        marker = "o" if prediction == 0 else "X"
        ax.scatter([features[0, 0]], [features[0, 1]], c="#E11D48", s=140, marker=marker, edgecolors="white", linewidths=1.5, label="Current Query Point", zorder=5)

        ax.set_xlabel(r"$X_1$ (Amp Diff, dB)", fontsize=9)
        ax.set_ylabel(r"$X_2$ (Loss Diff, dB)", fontsize=9)
        ax.set_xlim(-25, 10)
        ax.set_ylim(-35, 10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, linestyle=":", alpha=0.6)
        st.pyplot(fig)


# ==================== PAGE 2: MODEL BENCHMARKS ====================
elif page == "📊 Model Benchmarks & Comparison":
    st.markdown('<div class="main-header">Trained Model Comparison & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation on 1,500 hold-out test samples (25% split).</div>', unsafe_allow_html=True)

    metrics = artifacts["metrics"]

    # Metric Cards
    cols = st.columns(4)
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i % 4]:
            st.markdown(f"**{name.split('(')[0].strip()}**")
            st.metric("Test Accuracy", f"{m['Accuracy']*100:.2f}%")
            st.caption(f"Precision: {m['Precision']:.3f} | Recall: {m['Recall']:.3f}")
            st.caption(f"F1-Score: {m['F1-Score']:.3f}")

    st.markdown("---")

    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.subheader("Model Comparison Bar Chart")
        st.image("outputs/figures/model_comparison.png", use_container_width=True)

    with col_b:
        st.subheader("Decision Boundaries Comparison")
        st.image("outputs/figures/decision_boundaries.png", use_container_width=True)

    st.subheader("Confusion Matrices (1,500 Test Samples)")
    st.image("outputs/figures/confusion_matrices.png", use_container_width=True)

    st.markdown("#### Quantitative Metrics Summary Table")
    df_metrics = pd.DataFrame(metrics).T[["Accuracy", "Precision", "Recall", "F1-Score", "Intercept", "Coefficients"]]
    st.dataframe(df_metrics, use_container_width=True)


# ==================== PAGE 3: 19-SPAN TELEMETRY ====================
elif page == "🗺️ 19-Span Optical Telemetry":
    st.markdown('<div class="main-header">19-Span Optical Physical Telemetry Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of span-by-span optical line amplifier gain and fiber span attenuation.</div>', unsafe_allow_html=True)

    st.image("outputs/figures/span_loss_profile.png", use_container_width=True)

    st.markdown("""
    ### Physical Infrastructure Overview
    An end-to-end cloud backbone link spans hundreds of kilometers and is segmented into **19 discrete fiber spans**:
    - **Each Span:** Standard Single-Mode Fiber (SSMF, ~80 km) incurring $\\approx 0.2\\text{ dB/km} = 16.0\\text{ dB}$ nominal loss.
    - **Optical Amplifiers (EDFAs):** Placed at the termination of each span to compensate for fiber attenuation with nominal target gain $\\approx 16.0\\text{ dB}$.
    - **Physical Failure Modes:**
        1. **Fiber Attenuation Anomaly ($x_{2n} \\ll 0$):** Micro/macro-bending, connector contamination, aging splice degradation, or cable strain.
        2. **Amplifier Saturation / Pump Deficit ($x_{1n} \\ll 0$):** Aging laser diodes, pump laser breakdown, thermal runaway, or channel loading imbalance.
    """)


# ==================== PAGE 4: DATASET EXPLORATION ====================
elif page == "📂 Dataset Exploration":
    st.markdown('<div class="main-header">Dataset Characteristics & Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspection of the 6,000 optical link time-series dataset.</div>', unsafe_allow_html=True)

    st.image("outputs/figures/class_distribution.png", use_container_width=True)

    raw_path = Path("data/optical_network_raw.csv")
    feat_path = Path("data/optical_network_features.csv")

    if raw_path.exists() and feat_path.exists():
        raw_df = pd.read_csv(raw_path)
        feat_df = pd.read_csv(feat_path)

        st.subheader("Aggregated Features Preview (First 10 Rows)")
        st.dataframe(feat_df.head(10), use_container_width=True)

        st.write(f"**Total Samples:** {len(feat_df):,} rows")
        st.write(f"**Healthy Class (0):** {(feat_df['label'] == 0).sum():,} rows ({(feat_df['label'] == 0).mean()*100:.1f}%)")
        st.write(f"**Failing Class (1):** {(feat_df['label'] == 1).sum():,} rows ({(feat_df['label'] == 1).mean()*100:.1f}%)")


# ==================== PAGE 5: METHODOLOGY & ARCHITECTURE ====================
elif page == "📚 Methodology & Architecture":
    st.markdown('<div class="main-header">Methodology & Mathematical Formulations</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 1. Mathematical Feature Engineering (Eqs. 1 & 2)
    For each optical span $n \\in \\{0, 1, \\dots, 18\\}$:
    $$x_{1n} = \\mathrm{Ampgain}_n - \\mathrm{Targetgain}_n$$
    $$x_{2n} = \\mathrm{Target\\_span\\_loss}_n - \\mathrm{span\\_loss}_n$$

    Aggregated across all 19 spans:
    $$X_1 = \\sum_{n=0}^{18} x_{1n}, \\quad X_2 = \\sum_{n=0}^{18} x_{2n}$$

    ### 2. Evaluated Supervised Learning Classifiers
    - **Logistic Regression (Newton's Method):**
      $$J(\\theta) = -\\frac{1}{m}\\sum_{i=1}^m \\left[ y^{(i)}\\ln(h_\\theta(x^{(i)})) + (1-y^{(i)})\\ln(1-h_\\theta(x^{(i)})) \\right]$$
      $$\\theta \\leftarrow \\theta - H^{-1}\\nabla J(\\theta)$$
    - **Gaussian Discriminant Analysis (GDA):**
      Assumes joint distribution $p(x, y) = p(x|y)p(y)$ with shared covariance $\\Sigma$:
      $$\\theta = \\Sigma^{-1}(\\mu_1 - \\mu_0), \\quad \\theta_0 = \\frac{1}{2}(\\mu_0^T \\Sigma^{-1}\\mu_0 - \\mu_1^T \\Sigma^{-1}\\mu_1) + \\ln\\frac{\\phi}{1-\\phi}$$
      $$P(y=1|x) = \\sigma(\\theta^T x + \\theta_0)$$
    - **Support Vector Machine (Linear SVM):**
      $$\\min_{w, b, \\xi} \\frac{1}{2}\\|w\\|^2 + C\\sum_{i=1}^m \\xi_i \\quad \\text{s.t.} \\quad y^{(i)}(w^T x^{(i)} + b) \\ge 1 - \\xi_i$$
    """)
