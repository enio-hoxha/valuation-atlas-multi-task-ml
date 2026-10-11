import duckdb
import pandas as pd
import requests
import time
from tqdm import tqdm
from datetime import datetime
from unidecode import unidecode

def fetch_wiki_views_fuzzy(session, player_id, player_name, start_date_str):
    """
    Tries multiple URL variations to find the correct Wikipedia page.
    """
    end_date_str = datetime.now().strftime('%Y%m%d')
    
    # 1. Generate potential Wikipedia slugs
    name_undecoded = unidecode(player_name)
    slugs = [
        player_name.replace(" ", "_"),               # Exact name (Dávid_Hancko)
        f"{player_name.replace(' ', '_')}_(footballer)", # Disambiguation (Gavi)
        name_undecoded.replace(" ", "_"),            # Stripped accents (David_Hancko)
        f"{name_undecoded.replace(' ', '_')}_(footballer)" # Stripped + suffix
    ]
    
    # Remove duplicates but keep order
    slugs = list(dict.fromkeys(slugs))

    for slug in slugs:
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
                return data # Success
            elif response.status_code == 404:
                continue # Try next variant
        except Exception:
            continue
    return None

def run_recovery():
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    
    # Identify the Top 5000 players who are currently MISSING from our wiki table
    print("Identifying missing superstars...")
    missing_players = con.execute("""
        WITH target AS (
            SELECT player_id, name FROM players 
            WHERE market_value_in_eur IS NOT NULL 
            ORDER BY market_value_in_eur DESC LIMIT 5000
        )
        SELECT t.player_id, t.name 
        FROM target t
        LEFT JOIN (SELECT DISTINCT player_id FROM wikipedia_pageviews) w ON t.player_id = w.player_id
        WHERE w.player_id IS NULL
    """).df()

    if missing_players.empty:
        print("✅ No missing players in Top 5000 found!")
        return

    print(f"🚀 Found {len(missing_players)} missing players. Starting recovery...")

    session = requests.Session()
    session.headers.update({'User-Agent': 'FootballML-Project/1.0 (contact: your-email@example.com)'})

    recovered_count = 0
    for _, row in tqdm(missing_players.iterrows(), total=len(missing_players)):
        # We fetch from 2020 to ensure a full timeline for the ML model
        data = fetch_wiki_views_fuzzy(session, row['player_id'], row['name'], "20200101")
        
        if data:
            df_new = pd.DataFrame(data)
            con.execute("INSERT INTO wikipedia_pageviews SELECT * FROM df_new")
            recovered_count += 1
        
        # Respect rate limits (Wikimedia is generous, but 0.05 is safe)
        time.sleep(0.05)

    print(f"\n--- Recovery Complete ---")
    print(f"Successfully recovered {recovered_count} players.")
    con.close()

if __name__ == "__main__":
    run_recovery()