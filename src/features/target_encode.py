import pandas as pd
import numpy as np
import os
import json

def apply_target_encoding(train, val, test, id_mappings):
    """Calculate and apply Target Encoding for Clubs and Leagues."""
    
    # Merge mappings
    train = pd.merge(train, id_mappings, on='player_id', how='left')
    val = pd.merge(val, id_mappings, on='player_id', how='left')
    test = pd.merge(test, id_mappings, on='player_id', how='left')
    
    # Calculate target (Delta) for training set only
    train['target_delta'] = train['log_target_market_value_next_month'] - train['log_current_market_value']
    
    global_mean = train['target_delta'].mean()
    
    # Function for smoothed target encoding
    def get_smoothed_map(df, group_col, target_col, m=10):
        # m is the smoothing factor (min samples to trust the group mean)
        stats = df.groupby(group_col)[target_col].agg(['count', 'mean'])
        counts = stats['count']
        means = stats['mean']
        
        # Smoothed Mean = (count * group_mean + m * global_mean) / (count + m)
        smoothed = (counts * means + m * global_mean) / (counts + m)
        return smoothed.to_dict()

    # Calculate encodings on TRAIN
    club_premium_map = get_smoothed_map(train, 'club_id', 'target_delta', m=20)
    league_premium_map = get_smoothed_map(train, 'competition_id', 'target_delta', m=50)
    
    # Apply to all splits (using fillna(global_mean) for new/unknown IDs)
    for df in [train, val, test]:
        df['club_premium'] = df['club_id'].map(club_premium_map).fillna(global_mean)
        df['league_premium'] = df['competition_id'].map(league_premium_map).fillna(global_mean)
        
    # Drop IDs and intermediate target
    train = train.drop(columns=['club_id', 'competition_id', 'target_delta'])
    val = val.drop(columns=['club_id', 'competition_id'])
    test = test.drop(columns=['club_id', 'competition_id'])
    
    return train, val, test, club_premium_map, league_premium_map

if __name__ == "__main__":
    data_dir = "data/processed/ml_ready"
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    
    id_mappings = pd.read_csv('data/interim/id_mappings.csv')
    
    print("Applying Target Encoding for Club and League Premiums...")
    train, val, test, club_map, league_map = apply_target_encoding(train, val, test, id_mappings)
    
    # Save back
    train.to_parquet(f"{data_dir}/train.parquet")
    val.to_parquet(f"{data_dir}/val.parquet")
    test.to_parquet(f"{data_dir}/test.parquet")
    
    # Save maps for inspection
    model_dir = "models/features"
    os.makedirs(model_dir, exist_ok=True)
    # Filter out small counts or just save some samples
    with open(f"{model_dir}/premiums_meta.json", "w") as f:
        json.dump({
            "top_clubs": dict(sorted(club_map.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_leagues": dict(sorted(league_map.items(), key=lambda x: x[1], reverse=True)[:5])
        }, f, indent=4)
        
    print(f"✅ Target encoding applied. Features 'club_premium' and 'league_premium' added.")
