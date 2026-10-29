# Requirement Checklist and Project Audit

## Project: Link Failure Prediction and Localization in Cloud-Scale Networks Using Supervised Learning
**Course:** UE24CS352A - Machine Learning Mini-Project  
**Institution:** PES University  
**Assigned Paper:** "Link Failure Prediction and Localization in Cloud Scale Networks Using Supervised Learning" (Zahra Bakhtiari, Stanford University CS229)

---

## 1. Requirement Traceability Matrix

| ID | Requirement Category | Detailed Description | Source Reference | Mapped Implementation File(s) | Status |
|---|---|---|---|---|---|
| **REQ-01** | Workspace Audit | Inspect workspace, retain existing work, no deletion of functional components | Phase 1 Prompt | `README.md`, `deliverables/requirement_checklist.md` | Completed |
| **REQ-02** | Dataset Verification & Ingestion | Identify dataset provenance; support raw 19-span optical telemetry (4 axes: `span_loss`, `target_span_loss`, `amplifier_gain`, `target_gain`, OSNR target label, traffic direction); interpolate missing telemetry | Paper Sec. III, Phase 1 & 2 | `src/data_loader.py`, `src/preprocessing.py`, `data/` | Completed |
| **REQ-03** | Feature Engineering | Implement Eq. 1: $x_{1n} = Ampgain_n - Targetgain_n$, $x_{2n} = Target\_span\_loss_n - span\_loss_n$; Implement Eq. 2: aggregate across 19 spans: $X_1 = \sum_{n=0}^{18} x_{1n}$, $X_2 = \sum_{n=0}^{18} x_{2n}$ | Paper Sec. III Eq. 1-2 | `src/feature_engineering.py` | Completed |
| **REQ-04** | Fault Localization | Localize failing span ($0 \le n \le 18$) using maximum deviation in span loss / amplifier gain; transparently document when localization requires per-span telemetry vs aggregate link features | Paper Sec. I, III | `src/feature_engineering.py`, `app.py` | Completed |
| **REQ-05** | Split & Leakage Prevention | Strict 75% train / 25% test split ($N=6000$ points, 4500 train, 1500 test). Fit scalers/preprocessors exclusively on training split; identical split across all models | Paper Sec. III, IV; Phase 2 | `src/train.py`, `src/preprocessing.py` | Completed |
| **REQ-06** | Classifier 1: Logistic Regression | Standard Logistic Regression baseline optimized via Newton's Method; empirical loss minimization | Paper Sec. IV.1, Eq. 3 | `src/models/logistic_regression.py` | Completed |
| **REQ-07** | Classifier 2: Gaussian Discriminant Analysis (GDA) | Full Gaussian generative classifier ($\phi, \mu_0, \mu_1, \Sigma$), deriving posterior $P(y=1\|x)$ and linear decision boundary | Paper Sec. IV.2, Eq. 4 | `src/models/gda.py` | Completed |
| **REQ-08** | Classifier 3: Linear Support Vector Machine | SVM with linear kernel ($w^T x + b = 0$), comparing linear decision boundaries | Paper Sec. IV.3, Eq. 5 | `src/models/linear_svm.py` | Completed |
| **REQ-09** | Quantitative Evaluation | Evaluate Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report, Model Comparison Table | Paper Sec. V, Phase 3 | `src/evaluate.py`, `outputs/metrics/` | Completed |
| **REQ-10** | Visualizations | Generate high-res plots: Class distribution, Model comparison bar chart, Confusion matrices, 2D Scatter plot with linear decision boundaries ($P(y\|x)=0.5$) | Paper Figs. 3, 4, 5; Phase 3 | `src/evaluate.py`, `outputs/figures/` | Completed |
| **REQ-11** | Interactive Demonstration Application | Lightweight Streamlit web application: Dataset info, class distribution, comparison metrics, live inference, span localization, clear distinction between prediction vs physical network confirmation | Phase 4 | `app.py` | Completed |
| **REQ-12** | Test Suite & Quality Assurance | Modular automated tests covering preprocessing, feature engineering, model training, decision boundaries, localization, invalid inputs, and pipeline integration | Phase 5 | `tests/test_*.py` | Completed |
| **REQ-13** | Two-Page Project Summary Write-up | Concise 2-page academic PDF report covering problem statement, dataset details, methodology, implementation, and conclusions | Guidelines PDF Sec. 3.B | `deliverables/two_page_writeup.pdf`, `deliverables/generate_writeup_pdf.py` | Completed |
| **REQ-14** | Faculty Review Presentation Deck | Concise, polished presentation slides covering problem, architecture, feature engineering, models, results, demo, and viva prep | Guidelines PDF Sec. 4 | `deliverables/presentation_slides.md` | Completed |
| **REQ-15** | Live Demonstration Workflow | Structured demonstration script with sample test vectors and expected outputs | Guidelines PDF Sec. 4, Phase 6 | `deliverables/demo_workflow.md` | Completed |
| **REQ-16** | Viva Voce Preparation Guide | Comprehensive Q&A covering optical networking concepts, mathematical derivations (GDA, Newton's method, SVM), metrics, trade-offs, and project design | Guidelines PDF Sec. 4-5, Phase 6 | `deliverables/viva_preparation.md` | Completed |
| **REQ-17** | Environment & Reproducibility | Pinned dependencies (`requirements.txt`), complete setup and run commands in `README.md`, clean Git hygiene (`.gitignore`) | Phase 5 & 7 | `requirements.txt`, `README.md`, `.gitignore` | Completed |

---

## 2. Dataset Provenance and Gap Analysis

1. **Original Research Project Reference:**  
   The paper cited in `Problem statement.pdf` ("Link Failure Prediction and Localization in Cloud Scale Networks Using Supervised Learning" by Zahra Bakhtiari, Stanford CS229) lists the repository URL: `https://github.com/mehrnaz22/ML_project/blob/master/ML_Project.zip`.
2. **Current Network Audit:**  
   The remote GitHub repository `mehrnaz22/ML_project` returns `HTTP 404: Not Found` (either deleted or marked private by the original student author).
3. **Engineering Solution:**  
   - We implement a dual-mode data pipeline in `src/data_loader.py`.
   - **Mode A (User Data Ingestion):** If the user places `optical_network_raw.csv` or `optical_network_features.csv` in `data/`, the pipeline immediately ingests, validates, and runs on the raw telemetry.
   - **Mode B (Strict Mathematical Benchmark Simulation):** In the absence of an external CSV file, `data_loader.py` synthesizes the exact dataset described in Section III of the paper: 6,000 data points across 19 spans with the 4 raw time-series metrics (`span_loss`, `target_span_loss`, `amplifier_gain`, `target_gain`), realistic fiber attenuation ($0.2\text{ dB/km}$ across standard $80\text{ km}$ spans), EDFA gain saturation curves, linear interpolation for missing measurements, and ground truth OSNR degradation labeling.
   - All deviations or synthetic benchmark parameters are mathematically grounded in the paper's physical model and clearly reported.
