import pandas as pd
import os
import re
import duckdb

# Configuration
CSV_PATH = "data/scraped/capology/capology_salaries_20260427_125531.csv"
TEMP_CLEAN_CSV = "data/scraped/capology/capology_clean_temp.csv"

def slugify(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text).strip('-')
    return text

def extract_slug_from_url(url):
    if not isinstance(url, str) or not url:
        return None
    # Pattern: .../club/slug/salaries/...
    match = re.search(r'/club/([^/]+)/', url)
    if match:
        return match.group(1)
    return None

def ingest():
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Cleaning {len(df)} rows...")
    
    # 1. Clean Player Name
    df['player_name'] = df['name_clean']
    
    # 2. Extract or Generate Team Slug
    df['extracted_slug'] = df['club_url'].apply(extract_slug_from_url)
    df['team_slug'] = df['extracted_slug'].fillna(df['club_clean'].apply(slugify))
    
    # 3. Clean Weekly Wage
    def clean_currency(val):
        if pd.isna(val) or val == "" or val == "-":
            return 0.0
        if isinstance(val, str):
            val = re.sub(r'[^\d.]', '', val)
        try:
            return float(val)
        except:
            return 0.0

    df['weekly_wage_eur'] = df['weekly_gross_eur'].apply(clean_currency)
    
    # 4. Map Season
    df['season'] = df['season_name']
    
    # Final Selection
    final_df = df[['player_name', 'team_slug', 'weekly_wage_eur', 'season']]
    
    # Deduplicate
    initial_count = len(final_df)
    final_df = final_df.drop_duplicates(subset=['player_name', 'team_slug', 'season'])
    print(f"Dropped {initial_count - len(final_df)} duplicates.")
    
    # Save to temp CSV for DuckDB to pick up
    final_df.to_csv(TEMP_CLEAN_CSV, index=False)
    print(f"Saved cleaned data to {TEMP_CLEAN_CSV}")

    # Use DuckDB to push to Postgres
    print("Connecting to DuckDB and pushing to Postgres...")
    con = duckdb.connect()
    con.execute("INSTALL postgres;")
    con.execute("LOAD postgres;")
    
    # Attempt to connect to PG (assuming no password based on prior psql success)
    try:
        con.execute("ATTACH 'dbname=player_valuation user=postgres host=localhost' AS pg (TYPE postgres);")
        
        print("Truncating old data in Postgres and loading new records...")
        con.execute("DELETE FROM pg.capology_salaries") # Truncate equivalent in some versions
        con.execute(f"INSERT INTO pg.capology_salaries SELECT * FROM read_csv_auto('{TEMP_CLEAN_CSV}')")
        
        print(f"✅ Successfully ingested {len(final_df)} records into capology_salaries!")
    except Exception as e:
        print(f"❌ Error during Postgres push: {e}")
    finally:
        if os.path.exists(TEMP_CLEAN_CSV):
            os.remove(TEMP_CLEAN_CSV)
        con.close()

if __name__ == "__main__":
    if os.path.exists(CSV_PATH):
        ingest()
    else:
        print(f"Error: {CSV_PATH} not found.")
