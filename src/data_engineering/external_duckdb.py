import duckdb
import pandas as pd
import requests
import time
from tqdm import tqdm
from io import StringIO
from datetime import datetime, timedelta

def setup_external_tables(con):
    """Create tables only if they do not exist to allow for incremental updates."""
    con.execute("""
        CREATE TABLE IF NOT EXISTS clubelo_monthly (
            club TEXT,
            elo INTEGER,
            date DATE
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS wikipedia_pageviews (
            player_id INTEGER,
            player_name TEXT,
            date DATE,
            views INTEGER
        )
    """)

def fetch_clubelo(con, dates):
    """Fetch only missing ClubElo snapshots."""
    # Get dates already in DB
    existing_dates = con.execute("SELECT DISTINCT date FROM clubelo_monthly").df()['date'].tolist()
    # Convert to date objects for comparison
    existing_dates = [pd.to_datetime(d).date() for d in existing_dates]
    
    missing_dates = [d for d in dates if d.date() not in existing_dates]
    
    if not missing_dates:
        print("ClubElo data is already up to date.")
        return

    for date_obj in tqdm(missing_dates, desc="Updating ClubElo"):
        date_str = date_obj.strftime('%Y-%m-%d')
        url = f"http://api.clubelo.com/{date_str}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                df = pd.read_csv(StringIO(response.text))
                df["snapshot_date"] = date_obj.date()
                con.execute("""
                    INSERT INTO clubelo_monthly (club, elo, date)
                    SELECT Club, CAST(Elo AS INTEGER), CAST(snapshot_date AS DATE) FROM df
                """)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error ClubElo {date_str}: {e}")

def fetch_wiki_views(con, session, player_id, player_name, start_date_str):
    """Fetch Wiki views with a fuzzy fallback strategy."""
    end_date_str = datetime.now().strftime('%Y%m%d')
    
    # Hierarchy of slugs to try
    # 1. Raw name (e.g., Dávid_Hancko)
    # 2. Raw name + footballer (e.g., Gavi_(footballer))
    # 3. Unidecoded name (e.g., David_Hancko)
    slugs_to_try = [
        player_name.replace(" ", "_"),
        f"{player_name.replace(' ', '_')}_(footballer)",
        unidecode(player_name).replace(" ", "_")
    ]
    
    # Remove duplicates but maintain order
    slugs_to_try = list(dict.fromkeys(slugs_to_try))

    for slug in slugs_to_try:
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{slug}/monthly/{start_date_str}/{end_date_str}"
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                items = response.json().get("items", [])
                data = []
                for item in items:
                    data.append({
                        'player_id': player_id,
                        'player_name': player_name,
                        'date': pd.to_datetime(item['timestamp'][:6], format='%Y%m').date(),
                        'views': item['views']
                    })
                if data:
                    df_wiki = pd.DataFrame(data)
                    con.execute("INSERT INTO wikipedia_pageviews SELECT * FROM df_wiki")
                    return True # Success! Exit the slug loop
            elif response.status_code == 404:
                continue # Try the next slug in the list
        except:
            continue
    return False

def run_enrichment():
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    setup_external_tables(con)
    
    # 1. Update ClubElo
    print("Checking ClubElo for missing months...")
    target_dates = pd.date_range(start="2015-01-01", end=datetime.now(), freq="MS")
    fetch_clubelo(con, target_dates)
    
    # 2. Update Wikipedia (Incremental per player)
    print("Checking Wikipedia for missing months...")
    top_players = con.execute("""
        SELECT player_id, name 
        FROM players 
        WHERE market_value_in_eur IS NOT NULL 
        ORDER BY market_value_in_eur DESC 
        LIMIT 5000
    """).df()
    
    # Get the latest date we have for every player currently in the Wiki table
    last_dates = con.execute("SELECT player_id, MAX(date) as last_date FROM wikipedia_pageviews GROUP BY player_id").df()
    last_dates_dict = dict(zip(last_dates['player_id'], last_dates['last_date']))

    session = requests.Session()
    session.headers.update({'User-Agent': 'FootballML-Project/1.0'})

    current_month_start = datetime.now().replace(day=1).date()

    for _, row in tqdm(top_players.iterrows(), total=len(top_players), desc="Wiki Update"):
        pid = row['player_id']
        pname = row['name']
        
        # Determine when to start fetching
        if pid in last_dates_dict:
            last_date = pd.to_datetime(last_dates_dict[pid]).date()
            # If we already have data for the start of this month, skip
            if last_date >= current_month_start:
                continue
            # Start from the beginning of the month following our last data point
            # Format: YYYYMMDD
            next_month = (last_date + timedelta(days=32)).replace(day=1)
            start_str = next_month.strftime('%Y%m01')
        else:
            # New player? Fetch from our global start point
            start_str = "20200101"
        
        fetch_wiki_views(con, session, pid, pname, start_str)
        time.sleep(0.02)

    print("\n--- Incremental Update Complete ---")
    
    # Deduplicate one last time to be safe
    con.execute("""
        CREATE OR REPLACE TABLE clubelo_monthly AS 
        SELECT club, elo, date FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY club, date ORDER BY elo DESC) as rn
            FROM clubelo_monthly
        ) WHERE rn = 1
    """)
    
    con.execute("""
        CREATE OR REPLACE TABLE wikipedia_pageviews AS 
        SELECT player_id, player_name, date, views FROM (
            SELECT *, ROW_NUMBER() OVER(PARTITION BY player_id, date ORDER BY views DESC) as rn
            FROM wikipedia_pageviews
        ) WHERE rn = 1
    """)
    
    con.close()

if __name__ == "__main__":
    run_enrichment()