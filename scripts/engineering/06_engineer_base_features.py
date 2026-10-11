import pandas as pd
import numpy as np

def engineer_base_features(input_path, output_path):
    print(f"Loading panel from {input_path}...")
    # This panel already has wiki_popularity and club_strength from DuckDB
    # but might need the raw appearances data for rolling stats.
    # Actually, let's assume ingest.py created the monthly_panel with raw stats.
    # The merge_enriched script joined context.
    
    df = pd.read_parquet(input_path)
    
    print("Engineering base features...")
    # 1. Age (Static-ish but depends on month)
    if 'date_of_birth' in df.columns:
        df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
        df['age'] = (df['year_month'] - df['date_of_birth']).dt.days / 365.25
    
    # 2. Basic Stats (Already should be there from ingest.py)
    stat_cols = ['goals', 'assists', 'minutes_played', 'yellow_cards', 'red_cards']
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    
    # 3. Rolling averages (Lagged to avoid leakage)
    # We calculate them on the full set because they are purely historical
    print("Calculating rolling averages...")
    for col in stat_cols:
        # Shift(1) ensures we only use data BEFORE the current month
        df[f'{col}_3m'] = df.groupby('player_id')[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
    
    # 4. Targets
    # Market
    df['log_market_value'] = np.log1p(df['market_value_in_eur'])
    
    # Ensure chronological order for shifting
    df = df.sort_values(['player_id', 'year_month'])
    
    # Create a helper for strict next-month matching
    df['next_month_check'] = df.groupby('player_id')['year_month'].shift(-1)
    df['is_next_month'] = (df['next_month_check'] - df['year_month']).dt.days <= 31
    
    # Coach: Next month minutes (Strict)
    df['minutes_next_month'] = df.groupby('player_id')['minutes_played'].shift(-1)
    df.loc[~df['is_next_month'], 'minutes_next_month'] = np.nan
    
    # Agent: Value Change (next month) - Strict
    df['value_change'] = df.groupby('player_id')['market_value_in_eur'].diff().shift(-1)
    df.loc[~df['is_next_month'], 'value_change'] = np.nan
    
    # Sponsor: Wiki views next month - Strict
    if 'wiki_views' in df.columns:
        df['wiki_views_next_month'] = df.groupby('player_id')['wiki_views'].shift(-1)
        df.loc[~df['is_next_month'], 'wiki_views_next_month'] = np.nan
    else:
        df['wiki_views_next_month'] = 0

    # Cleanup helpers
    df = df.drop(columns=['next_month_check', 'is_next_month'])

    print(f"Saving engineered panel to {output_path}...")
    df.to_parquet(output_path)
    print(f"Engineered panel shape: {df.shape}")

if __name__ == "__main__":
    engineer_base_features('data/processed/final_enriched_panel.parquet', 'data/processed/engineered_spine.parquet')
