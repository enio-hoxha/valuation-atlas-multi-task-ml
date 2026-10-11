import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# The base URL for the raw GitHub files
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"

def download_file(url, filepath, session):
    """Helper function to download a file using a shared session."""
    if os.path.exists(filepath):
        return True # Skip if we already downloaded it
        
    try:
        # Use the shared session to get the file
        response = session.get(url, timeout=15)
        if response.status_code == 200:
            # Writing bytes ('wb') using response.content is slightly faster than text
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"  [!] Missing or error fetching: {url} (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"  [!] Connection error fetching {url}: {e}")
        return False

def main():
    filtered_file = "competitions_filtered.json"
    if not os.path.exists(filtered_file):
        print(f"Error: Could not find {filtered_file}. Please run the filter script first.")
        return

    with open(filtered_file, 'r', encoding='utf-8') as f:
        competitions = json.load(f)

    print(f"Found {len(competitions)} competitions to process.\n")

    # Ensure base directories exist
    for directory in ["data/matches", "data/events", "data/lineups"]:
        os.makedirs(directory, exist_ok=True)

    # We will gather all our download tasks into a list first
    download_tasks = []

    # Setup a shared session to reuse TCP connections
    with requests.Session() as session:
        
        # 1. Fetch match lists synchronously (since we need them to build the event URLs)
        for comp in competitions:
            c_id = str(comp["competition_id"])
            s_id = str(comp["season_id"])
            
            match_dir = os.path.join("data/matches", c_id)
            os.makedirs(match_dir, exist_ok=True)
            
            match_url = f"{BASE_URL}matches/{c_id}/{s_id}.json"
            match_filepath = os.path.join(match_dir, f"{s_id}.json")
            
            # Download the match index
            if download_file(match_url, match_filepath, session):
                with open(match_filepath, 'r', encoding='utf-8') as mf:
                    matches = json.load(mf)
                
                # Add all events and lineups for this competition to our task list
                for match in matches:
                    m_id = str(match["match_id"])
                    
                    download_tasks.append((
                        f"{BASE_URL}events/{m_id}.json", 
                        os.path.join("data/events", f"{m_id}.json")
                    ))
                    
                    download_tasks.append((
                        f"{BASE_URL}lineups/{m_id}.json", 
                        os.path.join("data/lineups", f"{m_id}.json")
                    ))

        print(f"Queued {len(download_tasks)} files for download. Starting threaded download...")

        # 2. Execute downloads concurrently using a ThreadPool
        # max_workers=10 downloads 10 files at a time. Be polite to GitHub and don't set this too high (10-20 is safe).
        successful_downloads = 0
        with ThreadPoolExecutor(max_workers=15) as executor:
            # Submit all tasks to the executor
            futures = {
                executor.submit(download_file, url, filepath, session): url 
                for url, filepath in download_tasks
            }
            
            # Process them as they complete
            for idx, future in enumerate(as_completed(futures)):
                if future.result():
                    successful_downloads += 1
                
                # Print progress every 100 files to keep the console clean
                if (idx + 1) % 100 == 0 or (idx + 1) == len(download_tasks):
                    print(f"  -> Progress: {idx + 1}/{len(download_tasks)} files processed.")

    print(f"\n✅ All {successful_downloads} data files downloaded successfully!")

if __name__ == "__main__":
    main()