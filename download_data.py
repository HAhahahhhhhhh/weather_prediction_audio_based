import requests
import json
import os

API_KEY = 'G4zP5ZJQNpnTNjBNX4enAdZbGRhxQyC36OWGaCCZ'
BASE_URL = 'https://freesound.org/apiv2'
SEARCH_TERMS = ['rain', 'windy', 'hail', 'thunder', 'snow']
OUTPUT_DIR = 'weather_audio_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_and_download(term, download_limit):
    page_size = 150  # Maximum page size allowed by FreeSound API
    downloaded_count = 0
    page = 1  # Start from the first page

    while downloaded_count < download_limit:
        url = f"{BASE_URL}/search/text/"
        params = {
            'query': term,
            'token': API_KEY,
            'fields': 'id,name,previews,license',
            'page_size': page_size,
            'page': page  # Specify the current page
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            results = response.json()
            for sound in results['results']:
                if downloaded_count >= download_limit:
                    break  # Stop if the download limit is reached
                download_audio(sound, term)
                downloaded_count += 1

            if 'next' not in results or not results['next']:
                # No more pages to fetch
                break
        else:
            print(f"Error with term '{term}': {response.status_code}")
            break
        page += 1  # Move to the next page


def download_audio(sound, label):
    try:
        preview_url = sound['previews']['preview-lq-mp3']
        file_name = f"{label}_{sound['id']}.mp3"
        file_path = os.path.join(OUTPUT_DIR, file_name)
        response = requests.get(preview_url)
        response.raise_for_status()  # Raise HTTPError for bad responses
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {file_name}")
    except Exception as e:
        print(f"Failed to download {sound['name']}: {e}")


for term in SEARCH_TERMS:
    search_and_download(term, 500)
