import pandas as pd
import numpy as np
import os

def add_valuation_pressure_features(train, val, test):
    """Bridge the forward-fill gap by tracking accumulated signal since the last update."""
    
    full_df = pd.concat([train, val, test]).sort_values(['player_id', 'month_date'])
    
    print("Detecting valuation update points and calculating pressure...")
    
    # 1. Identify where an actual update happened
    # (When current value is different from previous month's value)
    full_df['prev_market_value'] = full_df.groupby('player_id')['current_market_value'].shift(1)
    full_df['is_update_month'] = (full_df['current_market_value'] != full_df['prev_market_value']).astype(int)
    
    # Handle the first month for each player
    full_df.loc[full_df.groupby('player_id').head(1).index, 'is_update_month'] = 1
    
    # 2. Create a "Group ID" for periods between updates
    full_df['update_period_id'] = full_df.groupby('player_id')['is_update_month'].cumsum()
    
    # 3. Calculate "Pressure" (Cumulative stats within each update period)
    # This resets every time Transfermarkt updates the price
    full_df['goals_since_update'] = full_df.groupby(['player_id', 'update_period_id'])['monthly_minutes'].cumsum() # Proxy for activity
    
    # Better: Cumulative metrics
    metrics = {
        'monthly_minutes': 'minutes_since_update',
        'goals_p90': 'goals_accumulated',
        'wiki_views': 'hype_since_update'
    }
    
    for raw, new in metrics.items():
        full_df[new] = full_df.groupby(['player_id', 'update_period_id'])[raw].cumsum()
        
    # 4. "Time Pressure": How many months has it been since an update?
    full_df['months_since_last_update'] = full_df.groupby(['player_id', 'update_period_id']).cumcount()
    
    # Clean up
    cols_to_drop = ['prev_market_value', 'is_update_month', 'update_period_id']
    full_df = full_df.drop(columns=cols_to_drop)
    
    # Re-split
    new_train = full_df[full_df['month_date'] < '2023-01-01'].copy()
    new_val = full_df[(full_df['month_date'] >= '2023-01-01') & (full_df['month_date'] <= '2023-12-31')].copy()
    new_test = full_df[full_df['month_date'] >= '2024-01-01'].copy()
    
    return new_train, new_val, new_test

if __name__ == "__main__":
    data_dir = "data/processed/ml_ready"
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    
    print("Applying Valuation Pressure engineering...")
    train, val, test = add_valuation_pressure_features(train, val, test)
    
    # Save back
    train.to_parquet(f"{data_dir}/train.parquet")
    val.to_parquet(f"{data_dir}/val.parquet")
    test.to_parquet(f"{data_dir}/test.parquet")
    
    print(f"✅ Pressure features added. Now tracking accumulation since last update.")
