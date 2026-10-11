import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

def build_monthly_panel(db_path, output_path):
    con = duckdb.connect(db_path)
    
    print("Extracting competitions...")
    top5_leagues = ('ES1', 'FR1', 'GB1', 'IT1', 'L1')
    
    # 1. Get Players in Top 5 leagues
    print("Filtering players in Top-5 leagues...")
    players_query = f"""
    SELECT 
        p.player_id, 
        p.name, 
        p.current_club_id, 
        p.date_of_birth, 
        p.country_of_citizenship, 
        p.position, 
        p.sub_position, 
        p.foot, 
        p.height_in_cm,
        p.market_value_in_eur as current_market_value
    FROM players p
    JOIN clubs c ON p.current_club_id = c.club_id
    WHERE CAST(c.last_season AS INTEGER) >= 2020
    """
    players_df = con.execute(players_query).df()
    
    # 2. Get Appearances and aggregate monthly
    print("Aggregating monthly appearances...")
    appearances_query = f"""
    SELECT 
        a.player_id,
        a.date,
        a.goals,
        a.assists,
        a.minutes_played,
        a.yellow_cards,
        a.red_cards,
        g.competition_id
    FROM appearances a
    JOIN games g ON a.game_id = g.game_id
    WHERE g.competition_id IN {top5_leagues}
    """
    appearances_df = con.execute(appearances_query).df()
    appearances_df['date'] = pd.to_datetime(appearances_df['date'])
    appearances_df['year_month'] = appearances_df['date'].dt.to_period('M').dt.to_timestamp()
    
    monthly_stats = appearances_df.groupby(['player_id', 'year_month']).agg({
        'goals': 'sum',
        'assists': 'sum',
        'minutes_played': 'sum',
        'yellow_cards': 'sum',
        'red_cards': 'sum'
    }).reset_index()
    
    # 3. Get Player Valuations
    print("Extracting valuations...")
    valuations_query = """
    SELECT 
        player_id,
        date,
        market_value_in_eur
    FROM player_valuations
    """
    valuations_df = con.execute(valuations_query).df()
    valuations_df['date'] = pd.to_datetime(valuations_df['date'])
    valuations_df['year_month'] = valuations_df['date'].dt.to_period('M').dt.to_timestamp()
    
    # Take the last valuation of each month for each player
    monthly_valuations = valuations_df.sort_values('date').groupby(['player_id', 'year_month']).last().reset_index()
    
    # 4. Create the Panel (Cross Join player x months or just join existing data)
    # For simplicity and to avoid a massive sparse matrix, we join stats and valuations on player-month
    print("Joining data into monthly panel...")
    panel_df = pd.merge(monthly_stats, monthly_valuations[['player_id', 'year_month', 'market_value_in_eur']], 
                        on=['player_id', 'year_month'], how='outer')
    
    # Merge with player static info
    panel_df = pd.merge(panel_df, players_df, on='player_id', how='inner')
    
    # Sort and Fill
    panel_df = panel_df.sort_values(['player_id', 'year_month'])
    
    # Forward fill valuation (last known valuation)
    panel_df['market_value_in_eur'] = panel_df.groupby('player_id')['market_value_in_eur'].ffill()
    
    # Save
    print(f"Saving panel to {output_path}...")
    panel_df.to_pickle(output_path)
    print(f"Panel shape: {panel_df.shape}")

if __name__ == "__main__":
    build_monthly_panel('transfermarkt-datasets.duckdb', 'src/data/monthly_panel.pkl')
