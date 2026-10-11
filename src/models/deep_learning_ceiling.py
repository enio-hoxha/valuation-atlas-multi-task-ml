import pandas as pd
import numpy as np
import joblib
import os

def main():
    # Load test data and player metadata (need to join with original names if possible)
    # Since ml_ready might not have names, I'll try to find them in interim/player_crosswalk.csv or similar
    data_dir = "data/processed/ml_ready"
    test_df = pd.read_parquet(f"{data_dir}/test.parquet")
    
    # Load predictions generated in previous turns
    market_preds = np.load('temp_preds/market_preds.npy')
    nn_preds = np.load('temp_preds/nn_preds.npy')
    
    y_true = test_df['log_target_market_value_next_month'].values
    
    # Calculate Absolute Errors
    test_df['err_xgb'] = np.abs(y_true - market_preds)
    test_df['err_nn'] = np.abs(y_true - nn_preds)
    test_df['nn_win_margin'] = test_df['err_xgb'] - test_df['err_nn']
    
    # Identify cases where NN significantly beat XGB
    nn_wins = test_df[test_df['nn_win_margin'] > 0.5].copy()
    
    print(f"Total Test Samples: {len(test_df)}")
    print(f"Samples where NN was better than XGB: {len(test_df[test_df['nn_win_margin'] > 0]):,}")
    print(f"Samples where NN beat XGB by > 0.5 log-points: {len(nn_wins):,}")
    
    # Analyze clusters for NN wins
    if 'tactical_cluster' in test_df.columns:
        print("\nDistribution of NN 'Big Wins' by Tactical Cluster:")
        cluster_stats = nn_wins['tactical_cluster'].value_counts(normalize=True).head(5)
        print(cluster_stats)

    # Let's look for specific high-value or high-hype players
    # We need player names. I'll check if I can find a mapping.
    # Looking at directory structure: data/interim/player_crosswalk.csv
    try:
        names_df = pd.read_csv('data/interim/player_crosswalk.csv')
        # Assuming common 'player_id'
        nn_wins = nn_wins.merge(names_df[['player_id', 'pretty_name']], on='player_id', how='left')
    except:
        print("Could not load player names, using IDs.")

    print("\nTop 5 Instances where FT-Transformer out-performed XGBoost:")
    top_wins = nn_wins.sort_values(by='nn_win_margin', ascending=False).head(5)
    
    cols_to_show = ['player_id', 'nn_win_margin', 'err_xgb', 'err_nn', 'tactical_cluster']
    if 'pretty_name' in top_wins.columns:
        cols_to_show = ['pretty_name'] + cols_to_show
        
    print(top_wins[cols_to_show])

    # Summary Statistic: Ceiling Analysis
    avg_err_xgb = test_df['err_xgb'].mean()
    avg_err_nn = test_df['err_nn'].mean()
    
    print("\n" + "="*50)
    print("STATISTIC: THE DEEP LEARNING CEILING")
    print("="*50)
    print(f"Overall MAE XGBoost: {avg_err_xgb:.4f}")
    print(f"Overall MAE FT-Transformer: {avg_err_nn:.4f}")
    
    # Find a sub-segment where NN is better
    # Hypothesis: NN is better for high-hype players (wiki_views)
    high_hype = test_df[test_df['log_wiki_views'] > test_df['log_wiki_views'].quantile(0.9)]
    print(f"\nSub-segment: High Hype Players (Top 10% Wiki Views)")
    print(f"  XGB MAE: {high_hype['err_xgb'].mean():.4f}")
    print(f"  NN MAE:  {high_hype['err_nn'].mean():.4f}")
    
    if high_hype['err_nn'].mean() < high_hype['err_xgb'].mean():
        print(">>> Insight: FT-Transformer hierarchical learning excels in High-Hype scenarios.")
    else:
        print(">>> Insight: XGBoost remains robust even in High-Hype segments.")
    print("="*50)

if __name__ == "__main__":
    main()
