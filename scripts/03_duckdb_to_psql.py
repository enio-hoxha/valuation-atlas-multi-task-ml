import duckdb
con = duckdb.connect('transfermarkt-datasets.duckdb')
con.execute("INSTALL postgres;")
con.execute("LOAD postgres;")
con.execute("""
    ATTACH 'dbname=player_valuation user=postgres host=localhost password=your_password' 
    AS pg_db (TYPE postgres);
""")

# Push the raw tables
# Push the raw tables with a drop-if-exists safeguard
con.execute("DROP TABLE IF EXISTS pg_db.clubelo_monthly;")
con.execute("CREATE TABLE pg_db.clubelo_monthly AS SELECT * FROM clubelo_monthly;")

con.execute("DROP TABLE IF EXISTS pg_db.wikipedia_pageviews;")
con.execute("CREATE TABLE pg_db.wikipedia_pageviews AS SELECT * FROM wikipedia_pageviews;")

con.execute("DROP TABLE IF EXISTS pg_db.capology_salaries;")
con.execute("CREATE TABLE pg_db.capology_salaries AS SELECT * FROM capology_salaries;")

# Push the crosswalks
con.execute("DROP TABLE IF EXISTS pg_db.player_crosswalk;")
con.execute("CREATE TABLE pg_db.player_crosswalk AS SELECT * FROM player_crosswalk;")
con.execute("DROP TABLE IF EXISTS pg_db.club_crosswalk;")
con.execute("CREATE TABLE pg_db.club_crosswalk AS SELECT * FROM club_crosswalk;")

print("Pushing Transfermarkt valuations to Postgres...")
# Push the player_valuations table
con.execute("DROP TABLE IF EXISTS pg_db.player_valuations;")
con.execute("CREATE TABLE pg_db.player_valuations AS SELECT * FROM player_valuations;")

print("Pushing basic player metadata...")
# Also push the players table so you can join names to IDs later
con.execute("DROP TABLE IF EXISTS pg_db.players;")
con.execute("CREATE TABLE pg_db.players AS SELECT * FROM players;")

con.execute("DROP TABLE IF EXISTS pg_db.clubs;")
con.execute("CREATE TABLE pg_db.clubs AS SELECT * FROM clubs;")

con.execute("DROP TABLE IF EXISTS pg_db.competitions;")
con.execute("CREATE TABLE pg_db.competitions AS SELECT * FROM competitions;")

con.execute("DROP TABLE IF EXISTS pg_db.appearances;")
con.execute("CREATE TABLE pg_db.appearances AS SELECT * FROM appearances;")

con.execute("DROP TABLE IF EXISTS pg_db.games;")
con.execute("CREATE TABLE pg_db.games AS SELECT * FROM games;")

# 1. Create a sequence of months
con.execute("""
    CREATE OR REPLACE TABLE months AS 
    SELECT CAST(range AS DATE) as year_month
    FROM range(DATE '2020-01-01', CURRENT_DATE, INTERVAL 1 MONTH)
""")

con.close()
print("Success!")