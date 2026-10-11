import psycopg2
import pandas as pd
import warnings

# Silence the Pandas SQLAlchemy warning for this simple test script
warnings.filterwarnings('ignore', message='^pandas only supports SQLAlchemy connectable.*')

# Database Connection Config
DB_CONFIG = {
    "dbname": "player_valuation",
    "user": "postgres",
    "host": "localhost",
    "password": "your_password" # Update this if needed
}

class DataQualityTestSuite:
    def __init__(self):
        self.conn = psycopg2.connect(**DB_CONFIG)
        self.passed = 0
        self.failed = 0

    def run_test(self, test_name, category, query, expected_func):
        """Executes a SQL query and evaluates the result against an expected condition."""
        try:
            df = pd.read_sql_query(query, self.conn)
            is_valid, message = expected_func(df)
            
            if is_valid:
                print(f"✅ PASS | [{category}] {test_name}")
                self.passed += 1
            else:
                print(f"❌ FAIL | [{category}] {test_name}")
                print(f"   ↳ {message}")
                self.failed += 1
        except Exception as e:
            print(f"⚠️ ERROR | [{category}] {test_name}")
            print(f"   ↳ Execution failed: {str(e).strip()}")
            self.failed += 1
            # Reset transaction state so the next test can run if this one crashes
            self.conn.rollback() 
    
    def run_all_tests(self):
        print("="*60)
        print("🚀 RUNNING MECE DATA QUALITY TEST SUITE")
        print("="*60)

        # ==========================================================
        # 1. COMPLETENESS & COVERAGE (Is anything missing?)
        # ==========================================================
        self.run_test(
            "ClubElo Top 5 League Coverage (2024)",
            "Completeness",
            """
            SELECT comp.name AS league, COUNT(DISTINCT cc.elo_name) as team_count
            FROM competitions comp
            JOIN clubs c ON comp.competition_id = c.domestic_competition_id
            JOIN club_crosswalk cc ON c.club_id = cc.tm_club_id
            JOIN clubelo_monthly ce ON cc.elo_name = ce.club
            WHERE comp.competition_id IN ('GB1', 'ES1', 'IT1', 'L1', 'FR1')
              AND EXTRACT(YEAR FROM ce.date) = 2024
            GROUP BY comp.name;
            """,
            lambda df: (
                len(df) > 0 and df['team_count'].min() >= 18, 
                f"Some leagues have poor ClubElo coverage (Expected ~18-20):\n{df.to_string(index=False)}"
            )
        )

        self.run_test(
            "Top 100 Players Capology Coverage",
            "Completeness",
            """
            WITH top_100 AS (
                SELECT p.player_id, p.name 
                FROM players p
                JOIN player_valuations v ON p.player_id = v.player_id
                WHERE v.date >= '2024-01-01'
                ORDER BY v.market_value_in_eur DESC
                LIMIT 100
            )
            SELECT COUNT(pc.tm_player_id) as mapped_count
            FROM top_100 t
            LEFT JOIN player_crosswalk pc ON t.player_id = pc.tm_player_id AND pc.source = 'Capology';
            """,
            lambda df: (
                df.iloc[0]['mapped_count'] >= 80, 
                f"Only {df.iloc[0]['mapped_count']}/100 top players have Capology wage mappings."
            )
        )

        # ==========================================================
        # 2. UNIQUENESS & DEDUPLICATION (Is anything double-counted?)
        # ==========================================================
        self.run_test(
            "ClubElo Temporal Deduplication",
            "Uniqueness",
            """
            SELECT club, date, COUNT(*) as row_count
            FROM clubelo_monthly
            GROUP BY club, date
            HAVING COUNT(*) > 1;
            """,
            lambda df: (
                len(df) == 0, 
                f"Found {len(df)} duplicate club/date combinations in ClubElo."
            )
        )

        self.run_test(
            "Crosswalk 1:1 Enforcement (Clubs)",
            "Uniqueness",
            """
            SELECT tm_club_id, COUNT(elo_name)
            FROM club_crosswalk
            GROUP BY tm_club_id
            HAVING COUNT(elo_name) > 1;
            """,
            lambda df: (
                len(df) == 0, 
                f"Found {len(df)} Transfermarkt clubs mapped to multiple ClubElo teams."
            )
        )

        # ==========================================================
        # 3. RELATIONAL INTEGRITY (Are the bridges broken?)
        # ==========================================================
        self.run_test(
            "Orphaned Crosswalk Entries (Capology)",
            "Relational",
            """
            SELECT pc.external_name 
            FROM player_crosswalk pc
            LEFT JOIN capology_salaries cs ON pc.external_name = cs.player_name
            WHERE pc.source = 'Capology' AND cs.player_name IS NULL;
            """,
            lambda df: (
                len(df) == 0, 
                f"Found {len(df)} Capology crosswalk names that do not actually exist in the Capology table."
            )
        )

        # ==========================================================
        # 4. ATTRIBUTE VALIDITY (Does the data make logical sense?)
        # ==========================================================
        self.run_test(
            "No Negative Financials",
            "Validity",
            """
            SELECT 
                SUM(CASE WHEN market_value_in_eur <= 0 THEN 1 ELSE 0 END) as bad_valuations,
                SUM(CASE WHEN weekly_wage_eur <= 0 THEN 1 ELSE 0 END) as bad_wages
            FROM player_valuations v, capology_salaries s;
            """,
            lambda df: (
                df.iloc[0]['bad_valuations'] == 0 and df.iloc[0]['bad_wages'] == 0,
                f"Found negative/zero values. Valuations: {df.iloc[0]['bad_valuations']}, Wages: {df.iloc[0]['bad_wages']}"
            )
        )

        self.run_test(
            "Wage Plausibility (Top Earners)",
            "Validity",
            """
            SELECT MAX(weekly_wage_eur) as max_wage 
            FROM capology_salaries;
            """,
            lambda df: (
                df.iloc[0]['max_wage'] < 5000000, 
                f"Maximum wage is suspiciously high: {df.iloc[0]['max_wage']} EUR/week."
            )
        )

        # ==========================================================
        # 5. ML READINESS: TEMPORAL INTEGRITY (No Time Travel)
        # ==========================================================
        self.run_test(
            "Strict Panel Granularity (Appearances)",
            "ML Readiness",
            """
            SELECT player_id, date_trunc('month', date) as year_month, COUNT(game_id) as games_played
            FROM appearances
            GROUP BY player_id, year_month
            HAVING COUNT(game_id) > 10;
            """,
            lambda df: (
                len(df) == 0, 
                f"Temporal Leakage Risk: Found players with more than 10 games in a single month."
            )
        )

        self.run_test(
            "Valuation Frequency Bounds (Daily Duplicates)",
            "ML Readiness",
            """
            SELECT player_id, date, COUNT(*) as valuations_per_day
            FROM player_valuations
            GROUP BY player_id, date
            HAVING COUNT(*) > 1;
            """,
            lambda df: (
                len(df) == 0, 
                f"Found {len(df)} instances where a player has multiple different valuations on the EXACT SAME DAY."
            )
        )

        # ==========================================================
        # 6. ML READINESS: MATHEMATICAL BOUNDS (Target Validity)
        # ==========================================================
        self.run_test(
            "Poisson Target Bounds (Coach Model)",
            "ML Readiness",
            """
            SELECT 
                MIN(minutes_played) as min_mins,
                MAX(minutes_played) as max_mins
            FROM appearances;
            """,
            lambda df: (
                df.iloc[0]['min_mins'] >= 0 and df.iloc[0]['max_mins'] <= 150, 
                f"Poisson violation: Minutes played expected 0-150 per game. Found Min: {df.iloc[0]['min_mins']}, Max: {df.iloc[0]['max_mins']}"
            )
        )

        self.run_test(
            "ClubElo Target Bounds (Feature Scaling)",
            "ML Readiness",
            """
            SELECT MIN(elo) as min_elo, MAX(elo) as max_elo 
            FROM clubelo_monthly;
            """,
            lambda df: (
                df.iloc[0]['min_elo'] > 500 and df.iloc[0]['max_elo'] < 2500,
                f"ClubElo outside expected mathematical bounds (500-2500). Min: {df.iloc[0]['min_elo']}, Max: {df.iloc[0]['max_elo']}"
            )
        )

        self.run_test(
            "Biological Bounds (Age Feature)",
            "ML Readiness",
            """
            SELECT 
                MIN(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_of_birth))) as min_age,
                MAX(EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_of_birth))) as max_age
            FROM players
            WHERE date_of_birth IS NOT NULL;
            """,
            lambda df: (
                df.iloc[0]['min_age'] >= 14 and df.iloc[0]['max_age'] <= 60,
                f"Biological violation: Found players outside the 14-60 age range. Min: {df.iloc[0]['min_age']}, Max: {df.iloc[0]['max_age']}"
            )
        )

        # ==========================================================
        # SUMMARY
        # ==========================================================
        print("="*60)
        print(f"📊 TEST RUN COMPLETE: {self.passed} Passed, {self.failed} Failed.")
        print("="*60)

if __name__ == "__main__":
    tester = DataQualityTestSuite()
    tester.run_all_tests()