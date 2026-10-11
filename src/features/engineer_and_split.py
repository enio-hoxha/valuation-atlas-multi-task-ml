import pandas as pd
import numpy as np
import os
import duckdb
from tqdm import tqdm

def load_data_from_postgres():
    """Load the master panel from Postgres via DuckDB."""
    print("Connecting to Postgres and loading master_ml_panel_5...")
    con = duckdb.connect()
    con.execute("INSTALL postgres;")
    con.execute("LOAD postgres;")
    con.execute("ATTACH 'dbname=player_valuation user=postgres host=localhost' AS pg (TYPE postgres);")
    
    # Load into a dataframe
    df = con.execute("SELECT * FROM pg.master_ml_panel_5").df()
    con.close()
    return df

def engineer_features(df):
    """Apply log transforms, momentum features, and relative financials."""
    print(f"Engineering features for {len(df)} rows...")
    
    # 0. Ensure chronological order for grouping/filling/shifting
    df['month_date'] = pd.to_datetime(df['month_date'])
    df = df.sort_values(['player_id', 'month_date'])
    
    # 1. Forward Fill Sparse Financials (Player Level)
    # This prevents the "Zero Ground Truth" artifact by carrying over the last known value
    print("Performing player-level forward fills for market value and wages...")
    df['current_market_value'] = df.groupby('player_id')['current_market_value'].ffill()
    df['weekly_wage'] = df.groupby('player_id')['weekly_wage'].ffill()
    
    # Re-derive targets from filled values to recover missing ground truth
    df['target_market_value_next_month'] = df.groupby('player_id')['current_market_value'].shift(-1)
    df['target_wiki_views_next_month'] = df.groupby('player_id')['wiki_views'].shift(-1)
    
    # 2. Clean Position & Age
    df['age_at_obs'] = df['age_at_obs'].astype(float)
    
    # One-hot encode position
    if 'position' in df.columns:
        df = pd.get_dummies(df, columns=['position'], prefix='pos')
    
    # 3. Log Transformations (Handling right-skew)
    # Now targets will have far fewer NaNs/zeros
    financial_cols = ['weekly_wage', 'current_market_value', 'wiki_views', 
                      'target_market_value_next_month', 'target_wiki_views_next_month']
    for col in financial_cols:
        # Fill remaining start-of-career NaNs with 0 before log
        df[col] = df[col].fillna(0)
        df[f'log_{col}'] = np.log1p(df[col].astype(float))
    
    # 4. Relative Financials (Normalize against yearly inflation)
    df['year'] = df['month_date'].dt.year
    yearly_avg_wage = df[df['weekly_wage'] > 0].groupby('year')['weekly_wage'].transform('mean')
    df['relative_wage'] = df['weekly_wage'] / yearly_avg_wage.fillna(df['weekly_wage'].mean())
    
    # 5. Momentum Features (Rolling Deltas)
    # Elo Momentum (3-month % change)
    df['club_elo_3m_ago'] = df.groupby('player_id')['club_elo'].shift(3)
    df['elo_momentum_3m'] = (df['club_elo'] - df['club_elo_3m_ago']) / df['club_elo_3m_ago'].replace(0, np.nan)
    
    # MV Momentum (3-month % change)
    df['mv_3m_ago'] = df.groupby('player_id')['current_market_value'].shift(3)
    df['mv_momentum_3m'] = (df['current_market_value'] - df['mv_3m_ago']) / df['mv_3m_ago'].replace(0, np.nan)
    
    # 6. Hype Efficiency
    df['hype_to_value_ratio'] = df['wiki_views'] / df['current_market_value'].replace(0, np.nan)
    
    # 7. Missing Data Flags
    df['has_wage_data'] = (df['weekly_wage'] > 0).astype(int)
    
    # Drop intermediate columns
    df = df.drop(columns=['club_elo_3m_ago', 'mv_3m_ago', 'year'])
    
    return df

def split_and_save(df):
    """Perform OOT split and save to parquet."""
    print("Performing Out-Of-Time (OOT) splitting...")
    
    df['month_date'] = pd.to_datetime(df['month_date'])
    
    train = df[df['month_date'] < '2023-01-01']
    val = df[(df['month_date'] >= '2023-01-01') & (df['month_date'] <= '2023-12-31')]
    test = df[df['month_date'] >= '2024-01-01']
    
    output_dir = "data/processed/ml_ready"
    os.makedirs(output_dir, exist_ok=True)
    
    train.to_parquet(f"{output_dir}/train.parquet", index=False)
    val.to_parquet(f"{output_dir}/val.parquet", index=False)
    test.to_parquet(f"{output_dir}/test.parquet", index=False)
    
    print(f"✅ Data processed and saved to {output_dir}")
    print(f"   Train: {len(train)} rows")
    print(f"   Val:   {len(val)} rows")
    print(f"   Test:  {len(test)} rows")

if __name__ == "__main__":
    raw_df = load_data_from_postgres()
    engineered_df = engineer_features(raw_df)
    split_and_save(engineered_df)
