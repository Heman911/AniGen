import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from rapidfuzz import process, fuzz
import json
import os

# CACHE SETUP
CACHE_FILE = "image_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        image_cache = json.load(f)
else:
    image_cache = {}

# GLOBALS (lazy loaded)
df = None
similarity = None

def get_data():
    global df, similarity

    if df is None:
        df = pd.read_csv("data/anime_cleaned.csv")

        # CLEANING
        df.columns = df.columns.str.lower()
        df['title'] = df['title'].fillna("").astype(str).str.lower()

        # FIX GENRE COLUMN (SAFE)
        if 'genre' not in df.columns:
            if 'genre_x' in df.columns:
                df['genre'] = df['genre_x']
            elif 'genre_y' in df.columns:
                df['genre'] = df['genre_y']
            else:
                df['genre'] = ""
        else:
            df['genre'] = df['genre'].fillna("")

        # PROCESS GENRES
        df['genre_list'] = df['genre'].apply(
            lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
        )

        # FEATURE BUILDING
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(df['genre_list'])

        similarity = cosine_similarity(genre_matrix)

    return df, similarity

# FUZZY MATCH
def find_closest_title(anime_name):
    df, _ = get_data()
    titles = df['title'].tolist()

    match = process.extractOne(
        anime_name,
        titles,
        scorer=fuzz.partial_ratio
    )

    if match and match[1] > 50:
        return match[0]

    return None

# GET ANIME DATA (API)
def get_anime_data(title):
    global image_cache

    if title in image_cache:
        cached = image_cache[title]

        if isinstance(cached, str):
            return {
                "image": cached,
                "score": "N/A",
                "members": "N/A",
                "rank": "N/A"
            }

        return {
            "image": cached.get("image", "/static/default.png"),
            "score": cached.get("score", "N/A"),
            "members": cached.get("members", "N/A"),
            "rank": cached.get("rank", "N/A")
        }

    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        res = requests.get(url, timeout=5)

        if res.status_code != 200:
            raise Exception(f"API error: {res.status_code}")

        data = res.json()

        if data.get("data"):
            anime = data["data"][0]

            result = {
                "image": anime.get("images", {}).get("jpg", {}).get("image_url", "/static/default.png"),
                "score": anime.get("score", "N/A"),
                "members": anime.get("members", "N/A"),
                "rank": anime.get("rank", "N/A")
            }

            image_cache[title] = result

            with open(CACHE_FILE, "w") as f:
                json.dump(image_cache, f, indent=2)

            return result

    except Exception as e:
        print(f"[ERROR] Failed for {title}: {e}")

    return {
        "image": "/static/default.png",
        "score": "N/A",
        "members": "N/A",
        "rank": "N/A"
    }

# RECOMMENDER FUNCTION
def recommend_by_anime(anime_name, top_n=5):
    df, similarity = get_data()

    anime_name = anime_name.lower()
    matched_title = find_closest_title(anime_name)

    if not matched_title:
        return []

    idx = df[df['title'] == matched_title].index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []

    for i in scores:
        row = df.iloc[i[0]]
        title = row['title']

        if title != matched_title:
            data = get_anime_data(title)

            results.append({
                "title": title.title(),
                "image": data["image"],
                "score": data["score"],
                "members": data["members"],
                "rank": data["rank"],
                "genres": row.get('genre', 'N/A'),
                "type": row.get('type', 'N/A'),
                "episodes": row.get('episodes', 'N/A')
            })

        if len(results) == top_n:
            break

    return results

# TEST
if __name__ == "__main__":
    print(recommend_by_anime("naruto"))