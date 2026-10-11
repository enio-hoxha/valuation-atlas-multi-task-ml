# FootballML: Pipeline Execution Guide & Architecture Map

This directory contains the consolidated "Golden Path" of scripts for the FootballML valuation project. The scripts are numbered to reflect their logical execution order in the production pipeline.

## 1. Execution Sequence

| Step | Script | Phase | Role |
| :--- | :--- | :--- | :--- |
| **01** | `01_capology_scraper.py` | Collection | Asynchronous scraping of financial and salary history data. |
| **02** | `02_statbomb_fetch.py` | Collection | Retrieval of on-pitch event data and match statistics. |
| **03** | `03_duckdb_to_psql.py` | Ingestion | Synchronization of local analytical stores with the production PostgreSQL database. |
| **04** | `04_fuzzy_matcher.py` | Engineering | Resolving identity conflicts across datasets (e.g., mapping Capology names to StatBomb IDs). |
| **05** | `05_merge_enriched.py` | Engineering | Joining multi-source data into a unified temporal panel. |
| **06** | `06_engineer_base_features.py` | Features | Calculating rolling averages, lags, and biological attributes. |
| **07** | `07_unsupervised_signals.py` | Features | Generating tactical archetypes (K-Means) and anomaly scores (IsoForest). |
| **08** | `08_pressure_engineering.py` | Features | **Core Innovation:** Tracking accumulated performance since the last market value update. |
| **09** | `09_train_stakeholder_models.py` | Modeling | Training specialized XGBoost engines for Market, Coach, Sponsor, and Agent personas. |
| **10** | `10_train_deep_learning_model.py` | Modeling | SOTA FT-Transformer implementation for learning global feature interactions. |
| **11** | `11_train_meta_learner.py` | Ensemble | Ridge-based meta-learner that synthesizes all stakeholder views into a final valuation. |
| **12** | `12_final_system_evaluation.py` | Evaluation | Full-system validation on the 2024 test set (Established Assets). |
| **13** | `13_overfitting_analysis.py` | Evaluation | Gap analysis comparing Train vs. Val vs. Test performance. |

---

## 2. Conflicts & Superseded Logic

As the project evolved from a baseline regressor to a high-precision ensemble, several scripts were superseded. Users should prioritize the scripts in this directory over those in the root or `src/` folders.

*   **Monolithic vs. Persona:** `src/models/train.py` (MTL baseline) is superseded by `09_train_stakeholder_models.py`. The modular approach allows for persona-specific hyperparameter tuning.
*   **Total Value vs. Delta:** Early versions predicted total log-value. The current pipeline (`09` and `11`) focuses on the **Residual Delta**, forcing the model to explain *why* value changes rather than just anchoring to current price.
*   **Forward-Fill Correction:** Scripts not utilizing `08_pressure_engineering.py` will suffer from the "Static Month" bias where models cannot see performance signals between discrete market updates.

## 3. Configuration Utilities
*   `tuning_utility.py`: A standardized grid/random search harness for XGBoost optimization.
*   `target_encoding_utility.py`: Implements smoothed mean-encoding for high-cardinality features (Clubs/Leagues) to capture "Brand Premiums."
