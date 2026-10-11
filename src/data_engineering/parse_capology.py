import os
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
import duckdb

def parse_capology_html(html_path, team_slug):
    if not os.path.exists(html_path):
        return None
    
    with open(html_path, 'r') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    
    # Target all rows that have a data-index attribute, which usually identifies player rows in Capology's bootstrap tables
    rows = soup.find_all("tr", {"data-index": True})
    
    if not rows:
        # Fallback to searching for the name-column class
        rows = soup.find_all("tr")
    
    data = []
    for row in rows:
        cols = row.find_all("td")
        # According to debug output:
        # col 0: Name column (contains <a> with name)
        # col 2: Weekly Gross EUR (class money-column)
        if len(cols) >= 3:
            try:
                name_cell = row.find("td", class_="name-column")
                if not name_cell: continue
                
                link = name_cell.find("a")
                if not link: continue
                
                name = link.text.strip()
                # Clean name
                name = re.sub(r'\(.*\)', '', name).strip()
                
                # Find all money-columns in the row
                money_cols = row.find_all("td", class_="money-column")
                if not money_cols: continue
                
                # Usually the first money column is Weekly Gross EUR
                wage_str = money_cols[0].text.strip()
                
                # Clean wage: remove '€', ',', and other non-digits
                wage_clean = re.sub(r'[^\d]', '', wage_str)
                wage = float(wage_clean) if wage_clean else 0
                
                if wage > 100: # Threshold to filter out placeholder rows
                    data.append({
                        "player_name": name,
                        "team_slug": team_slug,
                        "weekly_wage_eur": wage,
                        "season": "2024-2025"
                    })
            except Exception as e:
                continue
    
    return pd.DataFrame(data)

def process_local_htmls():
    html_dir = 'data/raw/capology'
    all_data = []
    
    if not os.path.exists(html_dir):
        print(f"Directory {html_dir} not found.")
        return

    files = [f for f in os.listdir(html_dir) if f.endswith('.html')]
    for filename in tqdm(files, desc="Parsing HTMLs"):
        slug = filename.replace('.html', '')
        path = os.path.join(html_dir, filename)
        
        df = parse_capology_html(path, slug)
        if df is not None and not df.empty:
            all_data.append(df)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Deduplicate names within teams
        final_df = final_df.drop_duplicates(subset=['player_name', 'team_slug'])
        
        con = duckdb.connect('transfermarkt-datasets.duckdb')
        con.execute("CREATE OR REPLACE TABLE capology_salaries AS SELECT * FROM final_df")
        
        count = con.execute("SELECT COUNT(*) FROM capology_salaries").fetchone()[0]
        print(f"✅ Successfully loaded {count} unique salary records into DuckDB!")
        
        # Show Top 10 Earners in our scraped set
        print("\nTop 10 Earners in Scraped Dataset:")
        print(con.execute("SELECT player_name, team_slug, weekly_wage_eur FROM capology_salaries ORDER BY weekly_wage_eur DESC LIMIT 10").df())
        con.close()
    else:
        print("❌ No data parsed. Structure might still be mismatched.")

if __name__ == "__main__":
    process_local_htmls()
