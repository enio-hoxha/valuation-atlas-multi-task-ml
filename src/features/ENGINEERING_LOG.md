# Engineering Log: Feature Engineering & Preprocessing

## Phase 1: Feature Transformation & Splitting
**Date:** 2026-05-15

### Transformations Implemented:
0.  **Demographic Integration:**
    - Included `age_at_obs` (float) to capture the player's development stage.
    - One-hot encoded `position` (Attack, Midfield, Defender, Goalkeeper) to allow the model to learn position-specific valuation curves.
1.  **Logarithmic Scaling:**
    - Applied `Log1P` to `weekly_wage`, `wiki_views`, and `current_market_value` to handle extreme right-skewness in financial and attention data.
2.  **Relative Wage Calculation:**
    - `relative_wage`: `weekly_wage` divided by the annual mean of wages for that year. This isolates "financial importance" from general wage inflation.
3.  **Momentum Features:**
    - `elo_momentum_3m`: Percent change in `club_elo` over a 3-month rolling window.
    - `mv_momentum_3m`: Percent change in `current_market_value` over a 3-month rolling window.
4.  **Target Construction Verification:**
    - Confirmed `target_market_value_next_month` and `target_minutes_next_month` are properly shifted OOT.

### Data Splitting (OOT Strategy):
- **Train:** Observations where `month_date` < '2023-01-01'.
- **Validation:** Observations where `month_date` between '2023-01-01' and '2023-12-31'.
- **Test:** Observations where `month_date` >= '2024-01-01'.

### Missing Value Handling:
- Maintained `has_wiki_data` and created `has_wage_data` (1 if `weekly_wage > 0` else 0).
- Imputed `wiki_views` with 0 and `weekly_wage` with 0 before Log-scaling, relying on the dummy flags for model branching.
