import duckdb

con = duckdb.connect('transfermarkt-datasets.duckdb')
print("Deduplicating ClubElo...")

# Keep only the highest Elo rating if there are multiple fetches for the same club/date
con.execute("""
    CREATE TABLE clubelo_monthly_clean AS 
    SELECT club, date, MAX(elo) as elo 
    FROM clubelo_monthly 
    GROUP BY club, date;
""")
con.execute("DROP TABLE clubelo_monthly;")
con.execute("ALTER TABLE clubelo_monthly_clean RENAME TO clubelo_monthly;")

print("ClubElo is now deduplicated!")
con.close()