# ML Dataset Readiness Report: `master_ml_panel_5`

## 1. Overview
This report analyzes the readiness of the `master_ml_panel_5` materialized view for use in Multi-Task XGBoost training. This panel is restricted to the Top 5 European Leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) to ensure high signal-to-noise ratios.

## 2. High-Level Data Density (13-Season Spine)
*   **Total Observations:** 246,053 rows.
*   **Target Diversity:** Median Market Value ranges from **€3.0M to €8.0M**, capturing the elite professional tier.
*   **Financial Features (Capology Salaries):**
    *   **Post-2020 Coverage:** **~75%** match rate.
    *   **Historical Coverage (2013-2019):** **~54%** match rate.
*   **Commercial Features (Wikipedia Pageviews):**
    *   **Coverage:** **~70-75%** (2024-2026).
    *   **Temporal Availability:** High-fidelity data starts in **2020**.
*   **Activity Rate:** **100%**. Every row corresponds to active match participation in a Top 5 league.

## 3. Data Quality Analysis (Year-over-Year)

| Year | Obs Count | Unique Players | Avg Wage (EUR) | Median MV (EUR) | Wage Cov % | Wiki Cov % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2013 | 18,016 | 2,926 | 1,269,451 | 3,000,000 | 26.0% | 0.0% |
| 2016 | 17,932 | 2,961 | 1,440,676 | 3,500,000 | 54.3% | 0.0% |
| 2019 | 17,771 | 2,862 | 2,331,679 | 7,000,000 | 64.3% | 0.0% |
| 2022 | 17,836 | 2,975 | 2,614,947 | 6,000,000 | 74.3% | 45.1% |
| 2024 | 18,694 | 2,993 | 2,716,807 | 7,000,000 | 76.1% | 74.6% |

## 4. Preparation Strategy for ML Training

### A. Temporal Splitting (Out-Of-Time)
To prevent leakage and simulate real-world inference:
*   **Training:** Jan 2013 – Dec 2022.
*   **Validation:** Jan 2023 – Dec 2023.
*   **Testing:** Jan 2024 – Present.

### B. Feature Engineering Mandates
1.  **Relative Financials:** Normalize `weekly_wage` against the yearly average to account for market inflation.
2.  **Momentum Vectors:** Calculate 3-month and 6-month deltas for `club_elo` and `current_market_value`.
3.  **Hype Ratios:** Ratio of `wiki_views` to `current_market_value` to identify breakout potential.
4.  **Log-Scaling:** Apply `Log1P` to all currency and count-based features (`weekly_wage`, `wiki_views`, `current_market_value`).

### C. Imputation & Pre-processing
*   **Wage Imputation:** Use a flag-based approach (`has_wage_data`) to handle the ~25% missing values in the Top 5 leagues.
*   **Zero-Padding:** Tactical metrics (G/90, A/90) are already zero-padded for inactive months to maintain rolling average integrity.

---
*Report Generated: May 15, 2026*
