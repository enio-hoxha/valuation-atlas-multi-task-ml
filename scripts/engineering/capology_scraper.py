import asyncio
import json
import re
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
from datetime import datetime
import random
import sys

# Configuration
INPUT_JSON = "capology.com.2026-04-26T19_01_25.058Z.json"
OUTPUT_DIR = "data/scraped/capology"
CONCURRENCY_LIMIT = 1  # Sequential requests for maximum stability
DELAY_RANGE = (10.0, 20.0) # Longer randomized delay
LONG_BREATHER_EVERY = 5 # More frequent long breaks
LONG_BREATHER_TIME = 180 # Longer duration of long break

class CapologyScraper:
    def __init__(self, input_file):
        self.input_file = input_file
        self.semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        self.results = []
        self.processed_count = 0
        self.hit_403 = False
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )

    def fetch_sync(self, url):
        """Synchronous fetch using cloudscraper."""
        if self.hit_403:
            return None
        try:
            print(f"Fetching: {url}")
            response = self.scraper.get(url)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403:
                print(f"!!! Error 403 (Forbidden) encountered for {url} !!!")
                self.hit_403 = True
                return None
            else:
                print(f"Error {response.status_code} for {url}")
        except Exception as e:
            print(f"Exception for {url}: {e}")
        return None

    async def fetch(self, url):
        """Asynchronous wrapper for the synchronous scraper."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            html = await loop.run_in_executor(None, self.fetch_sync, url)
            if html:
                # Randomized delay to mimic human behavior
                await asyncio.sleep(random.uniform(*DELAY_RANGE))
            return html

    def parse_script_data(self, html, url):
        r"""Parses data from various var data(_\w+)? = [...] script blocks."""
        pattern = re.compile(r"var data(?:_\w+)?\s*=\s*\[(.*?)\];", re.DOTALL)
        matches = pattern.finditer(html)
        
        all_objects = []
        for match in matches:
            data_str = match.group(1)
            obj_pattern = re.compile(r"\{(.*?)\}", re.DOTALL)
            for obj_match in obj_pattern.finditer(data_str):
                obj_content = obj_match.group(1)
                entry = {}
                kv_pattern = re.compile(r"['\"](\w+)['\"]\s*:\s*(.*?),", re.DOTALL)
                for kv_match in kv_pattern.finditer(obj_content + ","):
                    key = kv_match.group(1)
                    value = kv_match.group(2).strip()
                    if value.startswith("'") or value.startswith('"'):
                        value = value[1:-1]
                    
                    if "accounting.formatMoney" in value:
                        money_match = re.search(r'["\']([\d\.\-]+)', value)
                        if money_match:
                            value = money_match.group(1)
                        else:
                            calc_match = re.search(r'["\']?([\d\.]+)', value)
                            if calc_match:
                                value = calc_match.group(1)
                    entry[key] = value
                
                if entry:
                    for field in ['name', 'season', 'verified', 'status', 'club']:
                        if field in entry and isinstance(entry[field], str) and "<" in entry[field]:
                            soup = BeautifulSoup(entry[field], "html.parser")
                            if field in ['name', 'season', 'club']:
                                a_tag = soup.find('a')
                                if a_tag:
                                    entry[f'{field}_text'] = a_tag.get_text()
                                    entry[f'{field}_url'] = urljoin(url, a_tag['href'])
                            entry[f'{field}_clean'] = soup.get_text()

                    # Extract league and season from URL
                    parts = url.strip('/').split('/')
                    try:
                        salaries_idx = parts.index('salaries')
                        entry['league'] = parts[salaries_idx - 1]
                        if salaries_idx + 1 < len(parts):
                            entry['season_name'] = parts[salaries_idx + 1]
                        else:
                            entry['season_name'] = "2025-2026" # Default for current
                    except (ValueError, IndexError):
                        entry['league'] = "unknown"
                        entry['season_name'] = "unknown"

                    entry['source_url'] = url
                    entry['scrape_date'] = datetime.now().isoformat()
                    all_objects.append(entry)
        return all_objects

    async def process_url(self, url):
        html = await self.fetch(url)
        if not html:
            return []

        data = self.parse_script_data(html, url)
        
        self.processed_count += 1
        if self.processed_count % 5 == 0:
            print(f"Processed {self.processed_count} High-Level URLs...")
            
        return data

    def get_already_scraped_urls(self):
        """Scans the output directory for existing CSVs and returns a set of already scraped source_urls."""
        scraped_urls = set()
        if not os.path.exists(OUTPUT_DIR):
            return scraped_urls
            
        for file in os.listdir(OUTPUT_DIR):
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(OUTPUT_DIR, file), usecols=['source_url'])
                    scraped_urls.update(df['source_url'].unique())
                except Exception as e:
                    pass
        return scraped_urls

    async def run(self):
        # Base URLs for Top 5 Leagues
        LEAGUE_BASES = [
            "https://www.capology.com/uk/premier-league/salaries/",
            "https://www.capology.com/es/la-liga/salaries/",
            "https://www.capology.com/it/serie-a/salaries/",
            "https://www.capology.com/de/1-bundesliga/salaries/",
            "https://www.capology.com/fr/ligue-1/salaries/"
        ]
        
        # Historical seasons from 2013-2014 to 2024-2025
        TARGET_SEASONS = [f"{year}-{year+1}" for year in range(2013, 2025)]
        
        target_urls = []
        for base in LEAGUE_BASES:
            target_urls.append(base) # Current season (2025-2026)
            for season in TARGET_SEASONS:
                target_urls.append(urljoin(base, f"{season}/"))

        already_scraped = self.get_already_scraped_urls()
        high_level_urls = [u for u in target_urls if u not in already_scraped]
        
        if not high_level_urls:
            print("All target URLs have already been scraped.")
            return

        print(f"Found {len(target_urls)} targets. Processing {len(high_level_urls)} NEW URLs...")
        
        chunk_size = 5
        for i in range(0, len(high_level_urls), chunk_size):
            if self.hit_403:
                print("Stopping due to 403 block. Saving collected data...")
                break
                
            chunk_urls = high_level_urls[i:i + chunk_size]
            tasks = [self.process_url(url) for url in chunk_urls]
            all_data_chunks = await asyncio.gather(*tasks)
            
            for data_chunk in all_data_chunks:
                if data_chunk:
                    self.results.extend(data_chunk)
            
            chunk_num = i//chunk_size + 1
            print(f"Chunk {chunk_num} completed. Total results: {len(self.results)}")
            
            # Long breather logic
            if chunk_num % LONG_BREATHER_EVERY == 0:
                print(f"Taking a long breather for {LONG_BREATHER_TIME} seconds...")
                await asyncio.sleep(LONG_BREATHER_TIME)
            else:
                await asyncio.sleep(5)

        if self.results:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(OUTPUT_DIR, f"capology_salaries_{timestamp}.csv")
            pd.DataFrame(self.results).to_csv(csv_path, index=False)
            print(f"Saved {len(self.results)} entries to {csv_path}")
            if self.hit_403:
                print("NOTE: Script stopped early due to 403. You can run it again later to resume.")
        else:
            print("No new data scraped.")

if __name__ == "__main__":
    scraper = CapologyScraper(INPUT_JSON)
    asyncio.run(scraper.run())
