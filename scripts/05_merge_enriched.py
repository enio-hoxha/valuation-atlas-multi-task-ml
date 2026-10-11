import duckdb
import pandas as pd
import os

def create_enriched_panel():
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    
    print("Building dense monthly spine (2020 - Present)...")
    
    # 1. Create a sequence of months
    con.execute("""
        CREATE OR REPLACE TABLE months AS 
        SELECT CAST(range AS DATE) as year_month
        FROM range(DATE '2020-01-01', CURRENT_DATE, INTERVAL 1 MONTH)
    """)
    
    # 2. Get active players (those with any appearance or valuation since 2020)
    con.execute("""
        CREATE OR REPLACE TABLE active_players AS
        SELECT DISTINCT player_id FROM (
            SELECT player_id FROM appearances WHERE date >= '2020-01-01'
            UNION
            SELECT player_id FROM player_valuations WHERE date >= '2020-01-01'
        )
    """)
    
    # 3. Create Dense Spine
    con.execute("""
        CREATE OR REPLACE TABLE dense_spine AS
        SELECT p.player_id, m.year_month
        FROM active_players p
        CROSS JOIN months m
    """)
    
    print("Merging enriched dataset onto dense spine...")
    
    query = """
    WITH monthly_stats AS (
        SELECT 
            player_id, 
            date_trunc('month', date) as year_month,
            SUM(minutes_played) as minutes_played,
            SUM(goals) as goals,
            SUM(assists) as assists,
            SUM(yellow_cards) as yellow_cards,
            SUM(red_cards) as red_cards
        FROM appearances
        GROUP BY player_id, year_month
    ),
    valuations_ffill AS (
        SELECT 
            player_id,
            date_trunc('month', date) as year_month,
            market_value_in_eur
        FROM player_valuations
    ),
    salaries_mapped AS (
        SELECT 
            pc.tm_player_id,
            cs.weekly_wage_eur
        FROM capology_salaries cs
        JOIN player_crosswalk pc ON cs.player_name = pc.external_name
        WHERE pc.source = 'Capology'
    )
    SELECT 
        s.player_id,
        p.name,
        p.date_of_birth,
        s.year_month,
        v.market_value_in_eur,
        ms.minutes_played,
        ms.goals,
        ms.assists,
        ms.yellow_cards,
        ms.red_cards,
        ce.elo as club_strength,
        wp.views as wiki_popularity,
        sm.weekly_wage_eur as weekly_wage
    FROM dense_spine s
    JOIN players p ON s.player_id = p.player_id
    LEFT JOIN monthly_stats ms ON s.player_id = ms.player_id AND s.year_month = ms.year_month
    
    -- Join Valuations (will need ffill in pandas)
    LEFT JOIN valuations_ffill v ON s.player_id = v.player_id AND s.year_month = v.year_month
    
    -- Join ClubElo via Fuzzy Crosswalk
    LEFT JOIN club_crosswalk ccw ON p.current_club_id = ccw.tm_club_id
    LEFT JOIN clubelo_monthly ce ON 
        ccw.elo_name = ce.club AND 
        s.year_month = date_trunc('month', ce.date)
    
    -- Join Wikipedia Views directly
    LEFT JOIN wikipedia_pageviews wp ON 
        s.player_id = wp.player_id AND 
        s.year_month = date_trunc('month', wp.date)
        
    -- Join Capology Salaries
    LEFT JOIN salaries_mapped sm ON
        s.player_id = sm.tm_player_id
    """
    
    df_enriched = con.execute(query).df()
    
    # Critical: Forward fill market value and other sparse signals per player
    print("Performing forward fills for sparse signals...")
    df_enriched = df_enriched.sort_values(['player_id', 'year_month'])
    df_enriched['market_value_in_eur'] = df_enriched.groupby('player_id')['market_value_in_eur'].ffill()
    df_enriched['weekly_wage'] = df_enriched.groupby('player_id')['weekly_wage'].ffill()
    
    # Fill remaining NaNs with defaults
    df_enriched['minutes_played'] = df_enriched['minutes_played'].fillna(0)
    df_enriched['goals'] = df_enriched['goals'].fillna(0)
    df_enriched['assists'] = df_enriched['assists'].fillna(0)
    df_enriched['club_strength'] = df_enriched['club_strength'].fillna(1500)
    df_enriched['wiki_popularity'] = df_enriched['wiki_popularity'].fillna(0)
    df_enriched['weekly_wage'] = df_enriched['weekly_wage'].fillna(0)
    
    # Drop rows before player's first ever appearance or valuation (to avoid excessive padding)
    print("Trimming excessive padding...")
    first_activity = df_enriched[df_enriched['market_value_in_eur'].notna() | (df_enriched['minutes_played'] > 0)]
    if not first_activity.empty:
        df_enriched = df_enriched[df_enriched['year_month'] >= first_activity['year_month'].min()]
    
    print(f"Final Enriched Shape: {df_enriched.shape}")
    os.makedirs('data/processed', exist_ok=True)
    df_enriched.to_parquet('data/processed/final_enriched_panel.parquet')
    print("Saved to data/processed/final_enriched_panel.parquet")
    con.close()

if __name__ == "__main__":
    create_enriched_panel()
