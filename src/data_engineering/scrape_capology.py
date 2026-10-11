import os
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
import duckdb

CLUB_MAPPING = {
    "Manchester City Football Club": "manchester-city",
    "Real Madrid Club de Fútbol": "real-madrid",
    "Futbol Club Barcelona": "barcelona",
    "FC Bayern München": "bayern-munich",
    "Paris Saint-Germain Football Club": "paris-saint-germain",
    "Liverpool Football Club": "liverpool",
    "Arsenal Football Club": "arsenal",
    "Chelsea Football Club": "chelsea",
    "Associazione Calcio Milan": "ac-milan",
    "Football Club Internazionale Milano S.p.A.": "inter-milan",
    "Juventus Football Club": "juventus",
    "Borussia Dortmund": "borussia-dortmund",
    "Club Atlético de Madrid S.A.D.": "atletico-madrid",
    "Tottenham Hotspur Football Club": "tottenham",
    "Manchester United Football Club": "manchester-united",
    "Società Sportiva Calcio Napoli": "napoli",
    "Bayer 04 Leverkusen Fußball": "bayer-leverkusen",
    "RasenBallsport Leipzig": "rb-leipzig",
    "Associazione Sportiva Roma": "roma",
    "Società Sportiva Lazio S.p.A.": "lazio"
}

def parse_capology_html(html_path, team_slug):
    if not os.path.exists(html_path):
        return None
    
    with open(html_path, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    table = soup.find("table")
    if not table:
        return None
    
    data = []
    rows = table.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            try:
                name_cell = cols[0]
                link = name_cell.find("a")
                if not link: continue
                
                name = link.text.strip()
                wage_str = cols[2].text.strip() # Gross Fixed P/W (EUR)
                
                # Clean wage
                wage_clean = re.sub(r'[^\d]', '', wage_str)
                wage = float(wage_clean) if wage_clean else 0
                
                if wage > 0:
                    data.append({
                        "player_name": name,
                        "team_slug": team_slug,
                        "weekly_wage_eur": wage,
                        "season": "2024-2025"
                    })
            except:
                continue
    
    return pd.DataFrame(data)

def run_firecrawl_scrape():
    os.makedirs('data/raw/capology', exist_ok=True)
    
    all_data = []
    # Limit to 5 for now
    mapping_subset = list(CLUB_MAPPING.items())[:5]
    for db_name, slug in tqdm(mapping_subset, desc="Scraping Capology"):
        output_file = f"data/raw/capology/{slug}.html"
        
        if not os.path.exists(output_file):
            url = f"https://www.capology.com/club/{slug}/salaries/"
            print(f"Scraping {slug}...")
            # Use firecrawl-cli via subprocess
            try:
                subprocess.run([
                    "firecrawl", "scrape", url, 
                    "--format", "html", 
                    "--wait-for", "5000", 
                    "-o", output_file
                ], check=True, capture_output=True)
            except Exception as e:
                print(f"Error scraping {slug}: {e}")
                continue
        
        df = parse_capology_html(output_file, slug)
        if df is not None:
            all_data.append(df)
            print(f"Parsed {len(df)} players for {slug}")

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        con = duckdb.connect('transfermarkt-datasets.duckdb')
        con.execute("CREATE OR REPLACE TABLE capology_salaries AS SELECT * FROM final_df")
        print(f"✅ Loaded {len(final_df)} salary records into DuckDB!")
        con.close()
    else:
        print("❌ No data found.")

if __name__ == "__main__":
    run_firecrawl_scrape()
