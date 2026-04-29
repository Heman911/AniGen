from flask import Flask, render_template, request, jsonify
from model.recommender import recommend_by_anime, get_anime_data
from rapidfuzz import process, fuzz
from model.classifier import predict_genres
import os
import pandas as pd

app = Flask(__name__)

# LAZY LOAD DATA
df = None

def get_data():
    global df
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

    return df

# HOME PAGE
@app.route('/')
def home():
    return render_template('index.html')

# AUTOCOMPLETE SEARCH
@app.route('/search')
def search():
    df = get_data()

    query = request.args.get('q', '').lower().strip()

    if not query:
        return jsonify([])

    titles = df['title'].dropna().tolist()

    starts_with = [t for t in titles if t.startswith(query)]
    contains_word = [t for t in titles if f" {query}" in t]

    fuzzy_matches = process.extract(
        query,
        titles,
        scorer=fuzz.token_sort_ratio,
        limit=10
    )

    fuzzy_titles = [match[0] for match in fuzzy_matches if match[1] > 65]

    results = starts_with + contains_word + fuzzy_titles
    results = list(dict.fromkeys(results))[:5]

    return jsonify(results)

# RECOMMEND BY ANIME
@app.route('/recommend', methods=['POST'])
def recommend():
    anime_name = request.form.get('anime_name')

    if not anime_name:
        return render_template('recommend.html', results=[])

    results = recommend_by_anime(anime_name)
    return render_template('recommend.html', results=results)

# RECOMMEND BY GENRE
@app.route('/recommend_by_genre', methods=['POST'])
def recommend_by_genre():
    df = get_data()

    genre = request.form.get('genre', '').lower()

    if not genre:
        return render_template('recommend.html', results=[])

    filtered = df[df['genre'].str.lower().str.contains(genre, na=False)]

    results = []

    for _, row in filtered.head(10).iterrows():
        data = get_anime_data(row['title'])

        results.append({
            "title": row['title'].title(),
            "image": data["image"],
            "score": data["score"],
            "members": data["members"],
            "rank": data["rank"],
            "genres": row.get('genre', 'N/A'),
            "type": row.get('type', 'N/A'),
            "episodes": row.get('episodes', 'N/A')
        })

    return render_template('recommend.html', results=results)

# CLASSIFIER
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        text = request.form.get('description')

        if not text or not text.strip():
            return render_template('classify.html', results=[])

        results = predict_genres(text)
        return render_template('classify.html', results=results)

    return render_template('classify.html', results=[])
# MINIGAMES
@app.route('/minigames')
def minigames():
    return render_template('minigames.html')

# DEBUG PANEL
@app.route('/debug')
def debug():
    df = get_data()
    sample = df.head(10).to_dict(orient='records')

    return {
        "columns": list(df.columns),
        "total_rows": len(df),
        "sample_data": sample
    }

# RUN (only for local)
if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)