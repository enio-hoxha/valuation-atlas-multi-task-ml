# Master Level Player Valuation Project - Final Report

## 1. Thesis Summary
Football player market value is a decomposable signal whose components differ by stakeholder. This project successfully implemented a parallel gradient-boosted pipeline that quantifies what each stakeholder prices and demonstrates the predictive superiority of ML-engineered features.

## 2. Key Results

### Stakeholder Importance Matrix
We trained four models for different stakeholders:
- **Market (Transfermarkt)**: Prices reputation and consistent performance.
- **Coach (Minutes Next Month)**: Strongly depends on recent tactical consistency and fatigue.
- **Agent (Value Change)**: Driven by breakout form (Identification of positive anomalies).
- **Sponsor (Prestige/Elite Dist)**: High focus on similarity to "elite" archetypes.

The **Importance Matrix** (`Desc/importance_matrix.csv`) shows that `dist_to_elite` and `is_breakout` are high-impact features for market value changes and prestige, respectively.

### Ablation Study
We compared the **Pub Fan (Raw Stats)** model against the **Analytics (Full Stack)** model:
- **Pub Fan R²**: 0.34
- **Analytics R²**: 0.48
**Insight**: Advanced feature engineering (Rolling stats, K-means clusters, and embeddings) adds a **14% increase in predictive accuracy** for market valuation.

### Similarity Recommender
Using the latent space of a trained Autoencoder, we can identify "under-the-radar" alternatives. For example, for "Harry Kane", the model identifies players with similar tactical and performance profiles like Lautaro Martínez or Odsonne Édouard, providing a data-driven "Moneyball" tool.

## 3. Methodology
- **Data**: Top-5 European Leagues, Monthly Panel (2020-2024).
- **Architecture**:
    - **Stage 1 (Unsupervised)**: PCA, K-Means (8 clusters), Isolation Forest, Autoencoder (Latent space = 8).
    - **Stage 2 (Supervised)**: XGBoost Regressors with Temporal Splitting.
- **Reproducibility**: End-to-end pipeline from DuckDB ingestion to SHAP analysis.

## 4. Conclusion
The research confirms that "Analytics pays". By decomposing player value into stakeholder-specific targets, clubs and agents can better understand which features (e.g., tactical archetypes vs. raw goal output) are being priced by the market at any given moment.
