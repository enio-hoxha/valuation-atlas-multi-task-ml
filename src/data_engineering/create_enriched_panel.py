import duckdb
import pandas as pd
import numpy as np
import os

def generate_final_panel():
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    
    print("Building temporal spine (2020-2026)...")
    
    con.execute("""
        CREATE OR REPLACE TABLE months AS 
        SELECT CAST(range AS DATE) as year_month
        FROM range(DATE '2020-01-01', DATE '2026-05-01', INTERVAL 1 MONTH)
    """)

    query = """
    WITH base_stats AS (
        SELECT 
            player_id, 
            date_trunc('month', date) as year_month,
            SUM(minutes_played) as minutes,
            SUM(goals) as goals,
            SUM(assists) as assists
        FROM appearances GROUP BY 1, 2
    ),
    wiki AS (
        SELECT player_id, date as year_month, views FROM wikipedia_pageviews
    ),
    elo AS (
        SELECT cc.tm_club_id, ce.date as year_month, ce.elo
        FROM club_crosswalk cc
        JOIN clubelo_monthly ce ON cc.elo_name = ce.club
    ),
    salaries AS (
        SELECT 
            pc.tm_player_id as player_id,
            cs.weekly_wage_eur,
            cs.season
        FROM capology_salaries cs
        JOIN player_crosswalk pc ON cs.player_name = pc.external_name
        WHERE pc.source = 'Capology'
    ),
    monthly_valuations AS (
        SELECT 
            player_id, 
            date_trunc('month', date) as year_month,
            arg_max(market_value_in_eur, date) as market_value_in_eur
        FROM player_valuations
        GROUP BY 1, 2
    )
    SELECT 
        p.player_id,
        p.name,
        m.year_month,
        v.market_value_in_eur,
        COALESCE(bs.minutes, 0) as minutes,
        COALESCE(bs.goals, 0) as goals,
        COALESCE(bs.assists, 0) as assists,
        COALESCE(e.elo, 1500) as club_strength,
        w.views as wiki_views,
        COALESCE(s.weekly_wage_eur, 0) as weekly_wage
    FROM (SELECT DISTINCT player_id FROM players WHERE market_value_in_eur > 0) active
    CROSS JOIN months m
    JOIN players p ON active.player_id = p.player_id
    LEFT JOIN base_stats bs ON active.player_id = bs.player_id AND m.year_month = bs.year_month
    LEFT JOIN monthly_valuations v ON active.player_id = v.player_id AND m.year_month = v.year_month
    LEFT JOIN wiki w ON active.player_id = w.player_id AND m.year_month = w.year_month
    LEFT JOIN elo e ON p.current_club_id::text = e.tm_club_id::text AND m.year_month = e.year_month
    LEFT JOIN salaries s ON active.player_id = s.player_id AND (
        (m.year_month >= CAST(SUBSTR(s.season, 1, 4) || '-07-01' AS DATE) AND 
         m.year_month <= CAST(SUBSTR(s.season, 6, 4) || '-06-01' AS DATE))
    )
    """

    print("Extracting to Pandas...")
    df = con.execute(query).df()

    print("Applying ML feature logic (Lagging Stats & Forward Fills)...")
    # Deduplicate in case multiple records for same season (take max)
    df = df.sort_values(['player_id', 'year_month', 'weekly_wage'])
    df = df.drop_duplicates(subset=['player_id', 'year_month'], keep='last')
    
    df = df.sort_values(['player_id', 'year_month'])
    df['market_value_in_eur'] = df.groupby('player_id')['market_value_in_eur'].ffill()
    df['weekly_wage'] = df.groupby('player_id')['weekly_wage'].ffill()
    
    cols_to_lag = ['market_value_in_eur', 'minutes', 'goals', 'assists', 'wiki_views', 'club_strength', 'weekly_wage']
    for col in cols_to_lag:
        df[col] = df.groupby('player_id')[col].shift(1)

    df['has_wiki_data'] = df['wiki_views'].notna().astype(int)
    df['wiki_views'] = df['wiki_views'].fillna(0)
    df['weekly_wage'] = df['weekly_wage'].fillna(0)
    df['log_popularity'] = np.log1p(df['wiki_views'])

    df = df.dropna(subset=['market_value_in_eur'])

    output_path = 'data/processed/final_enriched_panel.parquet'
    os.makedirs('data/processed', exist_ok=True)
    df.to_parquet(output_path)
    
    print(f"✅ Success! Enriched predictive panel saved to {output_path}")
    print(f"Total Rows: {len(df)}")
    con.close()

if __name__ == "__main__":
    generate_final_panel()
