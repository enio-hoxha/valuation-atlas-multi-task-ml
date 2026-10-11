import pandas as pd
from ScraperFC import Capology
import duckdb  # Optional: for DuckDB integration

# Initialize Capology scraper
capology = Capology()

# Define leagues and seasons (adjust as needed)
leagues = ["premier_league", "la_liga", "serie_a", "bundesliga", "ligue_1"]
seasons = range(2005, 2026)  # 2005–2025

# Fetch data for each league/season
dfs = []
for league in leagues:
    for season in seasons:
        try:
            print(f"Fetching {league} {season}-{season+1}...")
            # Use the correct method for ScraperFC v4.5+
            df = capology.get_data(league=league, season=season)
            if df is not None and not df.empty:
                df["league"] = league
                df["season"] = f"{season}-{season+1}"
                dfs.append(df)
        except Exception as e:
            print(f"Error fetching {league} {season}: {e}")

# Combine all data
if dfs:
    df_capology = pd.concat(dfs, ignore_index=True)
else:
    raise ValueError("No data fetched. Check league names and ScraperFC version.")

# Extract and rename key columns (adjust based on Capology's output)
df_capology = df_capology.rename(columns={
    "player": "player_name",
    "team": "team_name",
    "wage": "weekly_wage",
    "contract_years_remaining": "years_left_on_contract",
    "release_clause": "release_clause_indicator",
    "contract_expiry": "contract_expiry_date"
})

# Keep only relevant columns
df_capology = df_capology[[
    "player_name",
    "team_name",
    "weekly_wage",
    "years_left_on_contract",
    "release_clause_indicator",
    "contract_expiry_date",
    "league",
    "season"
]].dropna()

# Save to CSV (or DuckDB)
df_capology.to_csv("data/raw/capology_financial_data.csv", index=False)
print("Capology data saved to CSV!")

# Optional: Load into DuckDB
conn = duckdb.connect("football.db")
conn.execute("CREATE OR REPLACE TABLE capology_financial_data AS SELECT * FROM df_capology")
print("Capology data loaded into DuckDB!")