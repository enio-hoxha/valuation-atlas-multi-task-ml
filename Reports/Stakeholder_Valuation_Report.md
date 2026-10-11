# Stakeholder-Driven Valuation: Machine Learning Analysis Report

## 1. Executive Summary
This report details the implementation, optimization, and peak-performance analysis of a modular player valuation system. Moving beyond a monolithic valuation model, this project employs a **Stakeholder-Driven Architecture** that decomposes player value into four distinct dimensions: **Market (Valuation)**, **Coach (Utility)**, **Sponsor (Reach)**, and **Agent (Growth)**. By training independent, specialized XGBoost models for each persona, we provide a granular "Valuation Atlas" that aligns machine learning objectives with real-world football business requirements.

## 2. Methodology

### 2.1 Multi-Task vs. Independent Modeling
The project evolved through three distinct phases:
1.  **Baseline MTL:** A joint model for understanding target correlations.
2.  **Independent Stakeholder Models:** Specialized models for persona-specific drivers.
3.  **Residual Learning (Peak Optimization):** Shifting the Market target to the **Value Delta** to capture market velocity.

### 2.2 Solving the Valuation Lag: "Pressure Engineering"
A core challenge in transfer market data is the **Forward-Fill Gap**—Transfermarkt only updates prices 2-4 times a year, leaving many months "static." To solve this, we implemented **Valuation Pressure Engineering**:
*   **The Concept:** We treat current Market Value as "Accumulated Status" (Stock) and performance as "Incremental Pressure" (Flow).
*   **The Implementation:** Engineered features that track **Accumulated Goals, Minutes, and Hype since the last price update.** This allows the model to predict the "Burst" (the next re-valuation) even during static months.

## 3. Market Model Breakthrough: Peak R² Optimization

The "Market" persona underwent a surgical 4-stage optimization to maximize its explanatory power ($R^2$).

### 3.1 Optimization Strategies
*   **Target Encoding (Brand Premiums):** Captured "Club/League Brand" effects, quantifying the premium assigned to elite environments.
*   **Valuation Momentum (Lagged Deltas):** Integrated historical valuation trends (`lag1_delta`, `lag2_delta`) to detect price cycles.
*   **Pressure Accumulation:** Bridged the forward-fill gap by tracking performance since the last discrete value update.
*   **Logical Constraints:** Enforced **Monotonic Constraints** to ensure performance gains always translate to logical value increases.

### 3.2 Peak Performance Benchmarks (Market Model)
The final iteration achieved a breakthrough in explaining valuation variance, proving that "Predicting Change" is the real metric of success.

| Metric | Initial Baseline | Final Peak (Pressure-Aware) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Total Value R²** | 0.7464 | **0.7964** | **+6.71%** |
| **Delta R² (The "Change")** | 0.2213 | **0.3749** | **+69.41%** |
| **Test RMSE (Log)** | 1.2037 | **1.1682** | **+2.96%** |

---

## 4. Stakeholder Explainability (TreeSHAP)

Using TreeSHAP on the peak Market model, we identified the definitive drivers of valuation velocity.

### 4.1 Top Valuation Drivers (Global Importance)
1.  **Current Market Value (Gravity):** The primary anchor and resistor to delta change.
2.  **Value-per-Minute (Efficiency):** Distinguishes between starters and high-priced bench players.
3.  **ELO Momentum:** The dominant external catalyst; collective team success drives individual value.
4.  **Pressure Signals:** Accumulated output (goals/minutes) since the last update acts as the "trigger" for the next price jump.
5.  **Club Premium (Brand):** The inherent valuation "floor" provided by elite club status.

### 4.2 Institutional Learnings
*   **Status vs. Output:** The system confirms that a player's price is a mix of their **Legacy Status** (Market Value) and their **Current Pressure** (accumulated stats).
*   **The "Burst" Discovery:** By explaining 37.5% of the *change* in value, the model has moved from a simple "guesser" to a genuine **Transfer Market Simulator.**

## 5. Conclusion
The transition to a logically-constrained, pressure-aware residual model has yielded a high-precision valuation engine. With an **$R^2$ of 0.80**, the system provides a robust foundation for portfolio management, scouting for undervalued assets, and strategic transfer window decision-making.

---
*Technical Appendix: Models are stored in `models/residual/` and `models/independent/`.*
*Report Generated: May 15, 2026*
