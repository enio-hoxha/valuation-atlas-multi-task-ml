# valuation-atlas-multi-task-ml
Multi-Task &amp; Multi-View stacking ensemble (XGBoost, FT-Transformer, Ridge Meta-Learner) for alternative asset pricing and professional player valuation, validated on temporal Out-Of-Time data with DuckDB and Explainable AI (SHAP).
# Valuation Atlas: Multi-Task Ensemble Learning for Subjective Asset Pricing

[![Language](https://img.shields.io/badge/Language-Python-3776AB.svg?logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20XGBoost%20%7C%20DuckDB-EE4C2C.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-success)]()
[![Domain](https://img.shields.io/badge/Domain-Sports%20Economics%20%7C%20Alternative%20Assets-blue)]()
[![Methodology](https://img.shields.io/badge/Methodology-Multi--Task%20%7C%20Stacking%20Ensemble%20%7C%20XAI-orange)]()

A Multi-Task, Multi-View (MTL/MV) machine learning architecture designed to evaluate and forecast subjective pricing dynamics in alternative asset markets (professional football players across the Top 5 European Leagues, 2013-2024).

The system deconstructs monolithic valuation models by training specialized Level-0 proxy estimators targeting distinct industry stakeholder objectives, integrating connectionist tabular transformers (FT-Transformer in PyTorch), and aggregating predictions through a regularized Level-1 Ridge Regression Meta-Learner.

**Full Research Paper:** [Download Report PDF](docs/Valuation_Atlas_ML_Research_Report.pdf)

---

## Executive Summary and Architectural Core

- **Deconstructing Monolithic Valuation:** Traditional econometric models treat athlete market value as a single regression target, forcing algorithms to fit an arbitrary average. Valuation Atlas models conflicting stakeholder objective functions: Tactical Utility (Coach), Commercial Hype (Sponsor), Speculative Volatility (Agent), and Structural Market Value (Club).
- **Hybrid Learning Paradigms:** Precedes supervised modeling with unsupervised K-Means clustering (identifying topological tactical archetypes) and Isolation Forests (detecting performance breakout momentum).
- **Out-of-Time Generalization:** Validated on a strictly isolated 2024 Out-Of-Time (OOT) test block (n = 35,799). While specialized base models capture distinct idiosyncratic market signals, the terminal Ridge Meta-Learner achieves superior generalization performance (R-squared = 0.894, MAE = 0.152 log-points), outperforming standalone XGBoost baselines (R-squared = 0.851) and deep tabular architectures (R-squared = 0.872).
- **Explainable AI (XAI):** Global Tree-SHAP attributions and Partial Dependence Plots (PDP) uncover the macroeconomic gravity of the transfer market, verifying distinct model orthogonality and isolating non-linear biological asset depreciation curves.

---

## Multi-Task Stakeholder Formalization

The architecture evaluates three data views (Raw, Analytics, Engineered ML) against four specialized predictive targets:

1. **Coach Model (Tactical Utility):** Predicts expected minutes played at t+1 using **Poisson Regression** to capture non-negative count distributions and physical availability.
2. **Sponsor Model (Commercial Gravity):** Forecasts log-transformed Wikipedia pageviews at t+1 via **Root Mean Square Error (RMSE)** to model heavy-tailed digital media reach.
3. **Agent Model (Speculative Volatility):** Predicts absolute short-term market value deltas optimized via **Mean Absolute Error (MAE)** to isolate momentum spikes without sensitivity to stagnation.
4. **Market Model (Structural Baseline):** Predicts log market value using **Pseudo-Huber Loss**, an adaptive loss function that behaves quadratically (L2) for standard errors but linearly (L1) for record-breaking transfer outliers, preserving gradient stability.

---

## KDD Data Engineering Pipeline and Compute Infrastructure

```text
Raw Match Events & Valuations (StatsBomb & Transfermarkt)
        |
        +-- Financial Leverage Wages (Capology)
        +-- Digital Commercial Hype (Wikipedia API)
        +-- Environmental Context (ClubElo)
                    |
                    v  [DuckDB Analytical Engine]
Temporal Out-of-Core Joins & Parquet Materialization
                    |
                    v  [NLP Cleansing & Entity Resolution]
Fuzzy String Matching + History-Aware Registration Constraint
                    |
                    v  [Feature Engineering]
Target Encoding (Club IDs) | Exponential Youth Premium | Anomaly Scores
Relational Compute: Managed via DuckDB and columnar Parquet files, executing high-throughput out-of-core temporal joins across millions of event-level rows.

Entity Resolution: Resolved nominal inconsistencies across disparate schemas via an NLP normalization pipeline combining string cleaning, token dictionaries, fuzzy matching, and dual-condition validation (matching name similarity while strictly verifying simultaneous chronological club registration).

Structural Sparsity vs. Missingness: Missing wages handled via flag-based imputation; tactical metrics zero-padded during inactive periods to convert absence into a valid deflationary signal.

Temporal Window Lagging: Target variables lagged by one period (X_t evaluated against y_{t+1}) to strictly prevent look-ahead bias and data leakage.

Supervised Ensemble and Meta-Learning Pipeline
Input Feature Space (Tabular + Unsupervised Archetypes)
        |
        +--> Level-0: XGBoost Coach (Poisson Loss)
        +--> Level-0: XGBoost Sponsor (RMSE Loss)
        +--> Level-0: XGBoost Agent (MAE Loss)
        +--> Level-0: XGBoost Market (Pseudo-Huber Loss)
        +--> Level-0: FT-Transformer (Multi-Head Self-Attention, AdamW)
                    |
                    v  [Concatenated Predictions Matrix]
Level-1 Stacking: Ridge Regression Meta-Learner (L2 Shrinkage)
                    |
                    v
Consensus Market Valuation (R-squared = 0.894, MAE = 0.152)
The Level-1 Regularized Meta-LearnerConcatenating Level-0 continuous predictions into a Level-1 feature matrix creates severe multicollinearity. Unregularized linear regression in this space leads to ill-conditioned matrix inversion and unstable, high-variance coefficient estimates.The ensemble deploys a Ridge Regression Meta-Learner:$$\min_{\beta} \sum_{i=1}^{n} \left( y_i - \hat{y}_i^{\text{Level-0}} \beta \right)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$The L2 shrinkage parameter penalizes extreme coefficient magnitudes, compelling collinear base-learner predictions to share predictive influence proportionally. This stabilizes variance and delivers optimal out-of-sample consensus weighting.Empirical Results and Model ComparisonEvaluated on the vaulted Out-Of-Time (OOT) Test Block (2024-Present, n = 35,799):
Model Architecture,Paradigm / Objective,R-squared,MAE (Log-Points)
XGB_Agent,Proxy Model (MAE),0.112,1.250
XGB_Coach,Proxy Model (Poisson),0.154,1.120
XGB_Sponsor,Proxy Model (RMSE),0.321,0.890
XGB_Market,Structural Baseline (Pseudo-Huber),0.851,0.220
FT-Transformer,Tabular Deep Learning (Self-Attention),0.872,0.185
Ridge Meta-Learner,Stacked Generalization (Level-1),0.894,0.152

The specialized persona models capture orthogonal sub-components of value; their synthesis via the Ridge Meta-Learner achieves the highest predictive accuracy across the entire distribution.

Key Visualizations
1. Latent Tactical Space (K-Means Topological Continuum)
t-SNE projection of tactical archetypes replacing subjective nominal positions:

2. Comparative Generalization Performance (R-squared)
Out-of-Time test performance across base models, tabular transformer, and stacking meta-learner:

3. Actual vs. Predicted Density (Hexagonal Binning)
Hexbin concentration along the theoretical 45-degree line demonstrating absence of structural dispersion:

4. Stakeholder Matrix and Model Orthogonality
SHAP attribution heatmap confirming distinct inductive biases learned across Level-0 persona models:

5. Non-Linear Biological Depreciation Curve
Partial Dependence Plot (PDP) mapping the athletic lifecycle and peak valuation window (ages 25-27):

Repository Structure
├── docs/                                      # Academic report documentation
├── images/                                    # Plots and figures for README
├── scripts/                                   # Data processing and execution scripts
├── src/                                       # Core machine learning models and pipeline
└── README.md

Authors
Enio Hoxha - M.Sc. in Data Analytics for Economics and Management, Free University of Bozen-Bolzano
Lorenzo Casoni - M.Sc. in Data Analytics for Economics and Management, Free University of Bozen-Bolzano
