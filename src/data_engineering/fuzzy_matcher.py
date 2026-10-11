import pandas as pd
import duckdb
import json
import os
import glob
import re
from rapidfuzz import process, fuzz
from tqdm import tqdm
from unidecode import unidecode

def normalize(name):
    if not name: return ""
    # Remove accents, lowercase, and strip
    return unidecode(str(name)).lower().strip()

def slugify(text):
    if not text: return ""
    return normalize(text).replace(" ", "-").replace(".", "")

def normalize_club_name(name):
    """
    NLP Standardization layer to clean team names before fuzzy matching.
    Expands common aliases, applies ClubElo colloquialisms, and removes generic tokens.
    """
    if not name: return ""
    name = unidecode(str(name)).lower().strip()
    
    # --- The ClubElo Colloquial Dictionary ---
    # Maps Transfermarkt's formal names (or parts of them) to ClubElo's shorthand
    # --- The Ultimate ClubElo Colloquial Dictionary ---
    colloquial_map = {
        # BUNDESLIGA
        '1. fussball- und sportverein mainz 05': 'mainz',
        '1. fussball-club koln': 'koln',
        '1. fussballclub heidenheim 1846': 'heidenheim',
        '1. fussballclub union berlin': 'union berlin',
        'bayer 04 leverkusen fussball': 'leverkusen',
        'borussia dortmund': 'dortmund',
        'borussia verein fur leibesubungen 1900 monchengladbach': 'gladbach',
        'eintracht frankfurt fussball ag': 'frankfurt',
        'fc bayern munchen': 'bayern',
        'fussball-club augsburg 1907': 'augsburg',
        'fussball-club st. pauli von 1910': 'st pauli',
        'hamburger sport verein': 'hamburg',
        'rasenballsport leipzig': 'rb leipzig',
        'sport-club freiburg': 'freiburg',
        'sportverein werder bremen von 1899': 'werder bremen',
        'sv darmstadt 98': 'darmstadt',
        'turn- und sportgemeinschaft 1899 hoffenheim fussball-spielbetriebs': 'hoffenheim',
        'verein fur bewegungsspiele stuttgart 1893': 'stuttgart',
        'verein fur leibesubungen wolfsburg': 'wolfsburg',
        'vfl bochum': 'bochum',

        # LA LIGA
        'athletic club bilbao': 'athletic',
        'club atletico de madrid s.a.d.': 'atletico',
        'deportivo alaves s. a. d.': 'alaves',
        'elche club de futbol s.a.d.': 'elche',
        'futbol club barcelona': 'barcelona',
        'getafe club de futbol s. a. d. team dubai': 'getafe',
        'girona futbol club s. a. d.': 'girona',
        'levante union deportiva s.a.d.': 'levante',
        'rayo vallecano de madrid s. a. d.': 'rayo vallecano',
        'real betis balompie s.a.d.': 'betis',
        'real club celta de vigo s. a. d.': 'celta',
        'real club deportivo mallorca s.a.d.': 'mallorca',
        'real madrid club de futbol': 'real madrid',
        'real oviedo s.a.d.': 'oviedo',
        'real sociedad de futbol s.a.d.': 'sociedad',
        'real valladolid cf': 'valladolid',
        'reial club deportiu espanyol de barcelona s.a.d.': 'espanyol',
        'sevilla futbol club s.a.d.': 'sevilla',
        'valencia club de futbol s. a. d.': 'valencia',
        'villarreal club de futbol s.a.d.': 'villarreal',

        # LIGUE 1
        "angers sporting club de l'ouest": 'angers',
        'association de la jeunesse auxerroise': 'auxerre',
        'association sportive de monaco football club': 'monaco',
        'clermont foot 63': 'clermont',
        'football club de metz': 'metz',
        'football club de nantes': 'nantes',
        'football club lorient-bretagne sud': 'lorient',
        'le havre athletic club': 'le havre',
        'lille olympique sporting club': 'lille',
        'olympique de marseille': 'marseille',
        "olympique gymnaste club nice cote d'azur": 'nice',
        'olympique lyonnais': 'lyon',
        'paris football club': 'paris fc',
        'paris saint-germain football club': 'paris sg',
        'racing club de lens': 'lens',
        'racing club de strasbourg alsace': 'strasbourg',
        'stade brestois 29': 'brest',
        'stade reims': 'reims',
        'stade rennais football club': 'rennes',
        'toulouse football club': 'toulouse',

        # PREMIER LEAGUE
        'arsenal football club': 'arsenal',
        'association football club bournemouth': 'bournemouth',
        'aston villa football club': 'aston villa',
        'brentford football club': 'brentford',
        'brighton and hove albion football club': 'brighton',
        'burnley football club': 'burnley',
        'chelsea football club': 'chelsea',
        'crystal palace football club': 'crystal palace',
        'everton football club': 'everton',
        'fulham football club': 'fulham',
        'ipswich town': 'ipswich',
        'leeds united association football club': 'leeds',
        'liverpool football club': 'liverpool',
        'luton town': 'luton',
        'manchester city football club': 'man city',
        'manchester united football club': 'man united',
        'newcastle united football club': 'newcastle',
        'nottingham forest football club': 'forest',
        'sunderland association football club': 'sunderland',
        'tottenham hotspur football club': 'tottenham',
        'west ham united football club': 'west ham',
        'wolverhampton wanderers football club': 'wolves',

        # SERIE A
        'associazione calcio fiorentina': 'fiorentina',
        'associazione calcio milan': 'milan',
        'associazione sportiva roma': 'roma',
        'atalanta bergamasca calcio s.p.a.': 'atalanta',
        'bologna football club 1909': 'bologna',
        'cagliari calcio': 'cagliari',
        'calcio como': 'como',
        'football club internazionale milano s.p.a.': 'inter',
        'frosinone calcio': 'frosinone',
        'genoa cricket and football club': 'genoa',
        'juventus football club': 'juventus',
        'parma calcio 1913': 'parma',
        'pisa sporting club': 'pisa',
        'societa sportiva calcio napoli': 'napoli',
        'societa sportiva lazio s.p.a.': 'lazio',
        'torino calcio': 'torino',
        'udinese calcio': 'udinese',
        'unione sportiva cremonese s.p.a.': 'cremonese',
        'unione sportiva lecce': 'lecce',
        'unione sportiva sassuolo calcio': 'sassuolo',
        'us salernitana 1919': 'salernitana',
        'verona hellas football club': 'verona'
    }
    
    # Apply colloquial overrides if a formal name is found
    for formal, colloquial in colloquial_map.items():
        if formal in name:
            name = colloquial
            
    # 1. Expand standard football abbreviations
    name = re.sub(r'\bman\b', 'manchester', name)
    name = re.sub(r'\butd\b', 'united', name)
    name = re.sub(r'\bspurs\b', 'tottenham', name)
    
    # 2. Strip generic football tokens (FC, CF, SC, AFC, etc.)
    # \b ensures we only strip standalone words, not 'fc' inside other words
    generic_terms = r'\b(fc|cf|sc|afc|ud|rc|as|cd|de|fútbol|club)\b'
    name = re.sub(generic_terms, '', name)
    
    # 3. Clean up any double spaces left behind by removed tokens
    return " ".join(name.split())

def build_fuzzy_crosswalks():
    con = duckdb.connect('transfermarkt-datasets.duckdb')
    
    # =========================================================
    # 1. CLUB RESOLUTION: TM -> ClubElo
    # =========================================================
    print("\n--- Building Club Crosswalk (TM -> ClubElo) ---")
    tm_clubs_df = con.execute("SELECT DISTINCT club_id, name FROM clubs").df()
    elo_clubs = con.execute("SELECT DISTINCT club FROM clubelo_monthly").df()['club'].tolist()
    
    club_map = []
    # Use the new normalize_club_name for clubs
    norm_elo = [normalize_club_name(c) for c in elo_clubs]
    elo_dict = dict(zip(norm_elo, elo_clubs))
    
    for _, row in tqdm(tm_clubs_df.iterrows(), total=len(tm_clubs_df), desc="Matching Clubs"):
        tm_name = row['name']
        tm_id = row['club_id']
        norm_tm = normalize_club_name(tm_name)
        
        # Exact match check first (post-normalization)
        if norm_tm in elo_dict:
            club_map.append({'tm_club_id': tm_id, 'tm_name': tm_name, 'elo_name': elo_dict[norm_tm], 'score': 100})
            continue

        # Fuzzy match using token_set_ratio (handles "Real Madrid" vs "Real Madrid CF")
        match = process.extractOne(norm_tm, norm_elo, scorer=fuzz.token_set_ratio)
        
        # Because token_set_ratio is very forgiving, we enforce an 85 threshold
        if match and match[1] >= 85:
            original_elo_name = elo_dict[match[0]]
            club_map.append({'tm_club_id': tm_id, 'tm_name': tm_name, 'elo_name': original_elo_name, 'score': match[1]})
    
    df_club_crosswalk = pd.DataFrame(club_map)
    con.execute("DROP TABLE IF EXISTS club_crosswalk")
    con.execute("CREATE TABLE club_crosswalk AS SELECT * FROM df_club_crosswalk")
    print(f"Created club_crosswalk with {len(df_club_crosswalk)} matches.")

    # =========================================================
    # 2. PLAYER RESOLUTION: TM -> External Sources
    # =========================================================
    print("\n--- Building Player Crosswalk (TM -> External) ---")
    
    # Get TM players with club info (Remove last_season >= 2020 filter)
    tm_players_df = con.execute("""
        SELECT DISTINCT p.player_id, p.name, c.name as club_name
        FROM players p 
        JOIN clubs c ON p.current_club_id = c.club_id
    """).df()
    
    # We still use the standard `normalize` for players
    tm_players_df['norm_name'] = tm_players_df['name'].apply(normalize)
    tm_players_df['club_slug'] = tm_players_df['club_name'].apply(slugify)

    all_player_matches = []

    # A. Capology (Club-Aware)
    print("Matching Capology players (Club-Aware)...")
    cap_df = con.execute("SELECT DISTINCT player_name, team_slug FROM capology_salaries").df()
    
    for _, row in tqdm(cap_df.iterrows(), total=len(cap_df), desc="Capology"):
        cap_name = row['player_name']
        cap_team = row['team_slug']
        norm_cap_name = normalize(cap_name)
        
        # Filter TM players by club (exact slug match or fuzzy slug match)
        candidates = tm_players_df[tm_players_df['club_slug'].str.contains(cap_team.split('-')[0])]
        if candidates.empty:
            candidates = tm_players_df # Fallback to all if club not found
            
        # Match within candidates
        best_match = process.extractOne(norm_cap_name, candidates['norm_name'].tolist(), scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] > 85:
            tm_row = candidates[candidates['norm_name'] == best_match[0]].iloc[0]
            all_player_matches.append({
                'tm_player_id': tm_row['player_id'], 
                'tm_name': tm_row['name'], 
                'external_name': cap_name, 
                'source': 'Capology', 
                'score': best_match[1]
            })

    # B. StatsBomb & Wikipedia (Name-only fallback)
    def match_names_simple(source_names, source_label):
        matches = []
        tm_names_list = tm_players_df['norm_name'].tolist()
        for s_name in tqdm(source_names, desc=f"Matching {source_label}"):
            if not s_name: continue
            norm_s = normalize(s_name)
            match = process.extractOne(norm_s, tm_names_list, scorer=fuzz.token_sort_ratio)
            if match and match[1] > 90: # Higher threshold for name-only
                tm_row = tm_players_df[tm_players_df['norm_name'] == match[0]].iloc[0]
                matches.append({
                    'tm_player_id': tm_row['player_id'], 
                    'tm_name': tm_row['name'], 
                    'external_name': s_name, 
                    'source': source_label, 
                    'score': match[1]
                })
        return matches

    print("Matching StatsBomb & Wikipedia...")
    sb_names = set()
    lineup_files = glob.glob("StatsBomb/lineups/*.json")
    for f_path in lineup_files[:500]:
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                for team in data:
                    for player in team['lineup']:
                        sb_names.add(player['player_name'])
        except: continue
    
    all_player_matches.extend(match_names_simple(list(sb_names), "StatsBomb"))
    
    wiki_names = con.execute("SELECT DISTINCT player_name FROM wikipedia_pageviews WHERE player_id IS NULL").df()['player_name'].tolist()
    if wiki_names:
        all_player_matches.extend(match_names_simple(wiki_names, "Wikipedia"))

    df_player_crosswalk = pd.DataFrame(all_player_matches)
    con.execute("DROP TABLE IF EXISTS player_crosswalk")
    con.execute("CREATE TABLE player_crosswalk AS SELECT * FROM df_player_crosswalk")
    print(f"Created player_crosswalk with {len(df_player_crosswalk)} matches.")
    
    con.close()

if __name__ == "__main__":
    build_fuzzy_crosswalks()