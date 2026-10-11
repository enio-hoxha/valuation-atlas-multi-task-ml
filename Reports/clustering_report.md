# Clustering Analysis Report

## Existing Clustering (k=8)
- Features: ['goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 'age_at_obs', 'club_elo']
- Silhouette Score: 0.1569
- ARI (vs Position): 0.0056

## Refined Clustering (k=8)
- Features: ['goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m']
- Silhouette Score: 0.3768
- ARI (vs Position): 0.0195

## GMM Clustering (k=8)
- Features: ['goals_p90_roll_3m', 'assists_p90_roll_3m', 'minutes_roll_3m', 'age_at_obs', 'club_elo']
- Silhouette Score: -0.0362
- ARI (vs Position): 0.0295

## k-Evaluation (Original Features)
| k | Silhouette Score | ARI (vs Position) |
|---|------------------|-------------------|
| 4 | 0.2098 | 0.0085 |
| 5 | 0.2086 | 0.0085 |
| 6 | 0.2129 | 0.0087 |
| 7 | 0.2198 | 0.0080 |
| 8 | 0.2153 | 0.0062 |
| 9 | 0.2072 | 0.0082 |
| 10 | 0.2072 | 0.0069 |
| 11 | 0.2099 | 0.0071 |
| 12 | 0.2144 | 0.0071 |

## t-SNE Visualization
Plots saved to `../plots/`:
- `tsne_tactical_archetypes_publication.png`: Density-aware topological mapping of archetypes.
- `tsne_positions.png`: Player positions in t-SNE space for ground-truth comparison.
- `k_evaluation.png`: Silhouette and ARI across different k.
