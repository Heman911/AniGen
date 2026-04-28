import pandas as pd
import requests
import os
import re
from time import sleep

df = pd.read_csv("data/anime_full.csv")

os.makedirs("static/posters", exist_ok=True)

session = requests.Session()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "image/webp,*/*",
    "Referer": "https://myanimelist.net/"
}

for i, row in df.head(50).iterrows():  # keep small for now
    url = row.get("image_url")
    title = str(row.get("title"))

    title = re.sub(r'[^a-zA-Z0-9]', '_', title)
    filename = f"static/posters/{title}.jpg"

    if not url or os.path.exists(filename):
        continue

    try:
        response = session.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f" {title}")
        else:
            print(f" Status {response.status_code}: {title}")

    except Exception as e:
        print(f" Error: {title}")

    sleep(1)  # VERY IMPORTANT (avoid blocking)