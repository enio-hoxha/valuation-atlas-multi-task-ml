import pandas as pd
import numpy as np
import os

def add_lagged_deltas(train, val, test):
    """Calculate valuation momentum (previous month's deltas)."""
    
    # Combine to ensure we have the full history for lags
    # (Important for val/test rows that need lags from the end of the previous split)
    full_df = pd.concat([train, val, test]).sort_values(['player_id', 'month_date'])
    
    print("Calculating historical deltas and lags...")
    
    # Calculate the historical delta at each observation point
    # Historical Delta = Log(Current Value) - Log(Previous Month Value)
    full_df['prev_log_market_value'] = full_df.groupby('player_id')['log_current_market_value'].shift(1)
    full_df['hist_delta'] = full_df['log_current_market_value'] - full_df['prev_log_market_value']
    
    # Now create lags of that historical delta
    # lag1_delta: The change that happened LAST month
    full_df['lag1_delta'] = full_df.groupby('player_id')['hist_delta'].shift(1)
    # lag2_delta: The change that happened TWO months ago
    full_df['lag2_delta'] = full_df.groupby('player_id')['hist_delta'].shift(2)
    
    # Fill NaNs (for players at the start of their history)
    full_df['lag1_delta'] = full_df['lag1_delta'].fillna(0)
    full_df['lag2_delta'] = full_df['lag2_delta'].fillna(0)
    
    # Clean up intermediate cols
    full_df = full_df.drop(columns=['prev_log_market_value', 'hist_delta'])
    
    # Re-split based on original dates
    new_train = full_df[full_df['month_date'] < '2023-01-01'].copy()
    new_val = full_df[(full_df['month_date'] >= '2023-01-01') & (full_df['month_date'] <= '2023-12-31')].copy()
    new_test = full_df[full_df['month_date'] >= '2024-01-01'].copy()
    
    return new_train, new_val, new_test

if __name__ == "__main__":
    data_dir = "data/processed/ml_ready"
    train = pd.read_parquet(f"{data_dir}/train.parquet")
    val = pd.read_parquet(f"{data_dir}/val.parquet")
    test = pd.read_parquet(f"{data_dir}/test.parquet")
    
    print("Adding Lagged Deltas for momentum detection...")
    train, val, test = add_lagged_deltas(train, val, test)
    
    # Save back
    train.to_parquet(f"{data_dir}/train.parquet")
    val.to_parquet(f"{data_dir}/val.parquet")
    test.to_parquet(f"{data_dir}/test.parquet")
    
    print(f"✅ Lagged deltas added. Features 'lag1_delta' and 'lag2_delta' included.")
