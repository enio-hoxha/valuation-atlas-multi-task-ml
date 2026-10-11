import duckdb
import pandas as pd
import os

def extract_id_mappings(db_path, output_path):
    con = duckdb.connect(db_path)
    
    print("Extracting player-to-club and player-to-league mappings...")
    
    # 1. Player to Club (Static from current_club_id)
    query_club = """
    SELECT player_id, current_club_id as club_id
    FROM players
    """
    club_df = con.execute(query_club).df()
    
    # 2. Player to Primary Competition (Based on most appearances/minutes in the dataset)
    # This is a simplification, but helps capture the "League Brand"
    query_league = """
    SELECT player_id, competition_id
    FROM (
        SELECT 
            a.player_id, 
            g.competition_id, 
            ROW_NUMBER() OVER(PARTITION BY a.player_id ORDER BY COUNT(*) DESC) as rn
        FROM appearances a
        JOIN games g ON a.game_id = g.game_id
        GROUP BY a.player_id, g.competition_id
    ) WHERE rn = 1
    """
    league_df = con.execute(query_league).df()
    
    # Merge
    mapping_df = pd.merge(club_df, league_df, on='player_id', how='left')
    
    print(f"Saving mappings to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mapping_df.to_csv(output_path, index=False)
    con.close()

if __name__ == "__main__":
    extract_id_mappings('transfermarkt-datasets.duckdb', 'data/interim/id_mappings.csv')
