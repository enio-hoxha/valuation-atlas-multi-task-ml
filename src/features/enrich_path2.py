import pandas as pd
import numpy as np
import os

def add_path2_features(df):
    """Implement Feature Interaction & Signal Enrichment (Path 2)."""
    
    # 1. Interaction: Age + Club Strength
    # Younger players at stronger clubs often carry a higher valuation premium
    df['age_elo_interaction'] = (40 - df['age_at_obs']) * df['club_elo']
    
    # 2. Interaction: Performance + Value
    # This helps distinguish "expensive flops" from "underpriced gems"
    df['value_per_minute'] = df['log_current_market_value'] / (df['minutes_roll_6m'] + 1)
    
    # 3. Momentum: Team + Player
    # Does the player's value grow when the team is also on the rise?
    df['team_player_momentum'] = df['elo_momentum_3m'] * df['mv_momentum_3m']
    
    # 4. Hype Intensity
    # High wiki views relative to actual playing time (The "Wonderkid" effect)
    df['hype_intensity'] = df['log_wiki_views'] / (df['minutes_roll_6m'] + 1)
    
    # 5. Position Premiums (Interactions)
    # Market often values Attackers differently per age unit
    df['attacker_age_premium'] = df['pos_Attack'] * (30 - df['age_at_obs'])
    
    return df

if __name__ == "__main__":
    data_dir = "data/processed/ml_ready"
    
    for split in ['train', 'val', 'test']:
        path = f"{data_dir}/{split}.parquet"
        print(f"Enriching {split} data with Path 2 features...")
        df = pd.read_parquet(path)
        
        # Clean inf/nan before interactions
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        df = add_path2_features(df)
        
        # Save enriched version
        df.to_parquet(path)
        print(f"Saved {split}.parquet with {len(df.columns)} columns.")
