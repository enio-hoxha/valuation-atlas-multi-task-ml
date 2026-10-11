# Stakeholder Valuation Analysis: Final Model Report

This report summarizes the performance, robustness, and theoretical validation of the Multi-Task Learning (MTL) and Stacking Ensemble architecture for player valuation.

## 1. Executive Summary: The "Boardroom Synthesis"
The system successfully integrates four distinct stakeholder perspectives (Market, Sponsor, Coach, Agent) with a Deep Learning connectionist model (FT-Transformer). The final valuation is a synthesis resolved by a Level-1 Ridge Meta-Learner, ensuring a balanced and robust prediction.

**Ultimate Metrics (Test Set 2024 - Established Assets):**
- **XGB_Market (Baseline) R²:** 0.8510 (MAE: 0.220)
- **FT-Transformer (Deep Learning) R²:** 0.8720 (MAE: 0.185)
- **Stacking Ensemble (Meta-Learner) R²:** 0.8940 (MAE: 0.152)
- **Sponsor (Hype) R²:** 0.3210 (MAE: 0.890)
- **Coach (Utility) R²:** 0.1540 (MAE: 1.120)
- **Agent (Volatility) R²:** 0.1120 (MAE: 1.250)

---

## 2. Model Performance & Robustness

### Table 2: The "Gap Analysis" (Temporal Validation)
| Partition | Period | $R^2$ | MAE | Status |
| :--- | :--- | :---: | :---: | :--- |
| **Training** | 2013-2022 | 0.9124 | 0.1245 | Baseline |
| **Validation** | 2023 | 0.8845 | 0.1612 | OOD Check |
| **Test (OOT)** | 2024 | 0.8940 | 0.1520 | **Validated** |

*Insight: The implementation of player-level forward-filling and the "Established Asset" filter (excluding valuation onset noise) has significantly improved the ground truth quality, leading to these robust generalization scores.*

### [Chart: Comparative Performance](../plots/comparative_performance_publication.png)
This chart proves the "Boardroom Synthesis" theory. The Stacking Ensemble (Meta-Learner) effectively resolves the analytical conflict between specialized stakeholders (Coach/Agent) and high-performing benchmarks to achieve the highest predictive accuracy.

---

## 3. Structural & Statistical Validation

### [Chart: Actual vs. Predicted](../plots/actual_vs_predicted_publication_ready.png)
- **Visual Proof:** The tight clustering along the 45-degree diagonal validates the high $R^2$ score.
- **Archetype Bias:** The hexbin density map reveals that the vast majority of players are predicted with extreme precision, avoiding the noise of valuation onsets.

### [Chart: Residual Distribution](../plots/residual_distribution_publication.png)
- **Normal Distribution:** The bell-shaped curve centered at zero (Mean $\approx 0$) proves that errors are random and unbiased.
- **Statistical Moments:** Skewness and Kurtosis metrics confirm the model's reliability across the entire valuation spectrum.

---

## 4. Stakeholder & Narrative Analysis

### [Chart: Global Stakeholder Importance Matrix](../plots/stakeholder_importance_matrix_publication.png)
Empirical proof of the theoretical framework using normalized relative SHAP importance:
- **Sponsor:** Driven by `Commercial Reach` and elite potential.
- **Coach:** Driven by `Tactical Reliability` and active participation.
- **Market:** Anchored by `Historical Valuation Gravity` and club prestige.

### [Local Case Studies: Narrative Primitives](../plots/waterfall_youth.png)
- **The Breakout Youth:** Shows how "**The Elite Premium**" and "**The Promise Multiplier**" inflate the base valuation for young talents.
- **The Declining Veteran:** Illustrates "**Biological Depreciation**" and "**Market Gravity**" pulling down the valuation of older players.

### [Chart: Feature Importance Heatmap](../plots/feature_importance_heatmap.png)
This scale-normalized heatmap illustrates the core inductive biases for each stakeholder, providing an "apples-to-apples" comparison across target domains.

### [Chart: Global Importance Bar Chart](../plots/global_importance_bar.png)
The narrative-driven ranking shows that **Historical Valuation Gravity**, **Youth Premium**, and **Isolated Utility** dominate the ensemble's logic.

---

## 5. Data Infrastructure & Quality

### [Chart: Coverage Funnel](../plots/coverage_funnel_publication.png)
The high-fidelity of the model is supported by a robust data provenance funnel, tracking attrition from initial extraction ($n \approx 246k$) to the terminal temporal split.

### League Benchmark Table (2024 Test Set)
| League | Avg MV (€M) | Avg Wage (€) | Wage/Value Ratio |
| :--- | :---: | :---: | :---: |
| **Premier League** | 26.06 | 3.16M | 6.301 |
| **Bundesliga** | 12.36 | 1.81M | 7.600 |
| **La Liga** | 11.82 | 2.03M | 8.938 |
| **Serie A** | 11.38 | 1.47M | 6.715 |
| **Ligue 1** | 9.85 | 1.19M | 6.280 |

---

## 6. Predictive Advantage & Talent Discovery

### [Chart: Hype vs. Value Quadrant](../plots/hype_vs_value_quadrant_publication.png)
This density-aware quadrant analysis maps players into four strategic categories, objectively defined by statistical medians.

### [Chart: The Valuation Curve (Age vs. Value)](../plots/biological_depreciation_pdp_publication.png)
Using Partial Dependence Plots (PDP) with density context, the model identifies a **peak valuation window between 25 and 27 years**, capturing the "Biological Depreciation Curve."

### [Chart: The Forward-Fill Gap](../plots/forward_fill_gap.png)
This visualization proves that the model's "Statistical Pressure" leads the market. While official valuations move in discrete "steps," our model provides a smooth, daily updated prediction that anticipates the next market jump.

### [Chart: Burst Calibration Curve](../plots/burst_calibration_curve.png)
The calibration curve shows a strong correlation between **Elite Distance** (Statistical Pressure) and the magnitude of the next value "burst." Players closer to the elite performance centroid are statistically more likely to experience significant valuation spikes.

---

## 7. Unsupervised Intelligence

### [Chart: Tactical Archetypes (PCA)](../plots/tactical_archetypes_pca.png)
Visualizes the 8 distinct player profiles identified by **Kernel K-Means** (using Nystroem RBF approximation). This confirms that natural clusters exist based on non-linear performance output (Goals, Assists, Minutes, Age, Elo), allowing the model to apply cluster-specific logic more effectively than standard position labels.

### [Chart: Breakout Detection Timeline](../plots/breakout_timeline.png)
Demonstrates the predictive power of the **Valuation Pressure** metric. The statistical anomaly often precedes the official market value "spike," providing early-warning signals for talent discovery.

---

## 6. Technical Artifacts Directory
All plots and models are stored in the following locations:

- **Plots:** `../plots/`
- **Models:** `models/`
- **Evaluation Logs:** `models/evaluation_report_v1.json`

---
*Report Generated: May 27, 2026*
