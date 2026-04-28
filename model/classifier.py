import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# GLOBALS (lazy loaded)
df = None
tfidf = None
mlb = None
model = None

def get_model():
    global df, tfidf, mlb, model

    if model is None:
        df = pd.read_csv("data/anime_cleaned.csv")

        # CLEAN
        df.columns = df.columns.str.lower()
        df['synopsis'] = df['synopsis'].fillna("")

        # FIX GENRE
        if 'genre_x' in df.columns:
            df['genre'] = df['genre_x']
        elif 'genre_y' in df.columns:
            df['genre'] = df['genre_y']
        else:
            df['genre'] = ""

        df['genre'] = df['genre'].fillna("")
        df['genre'] = df['genre'].apply(
            lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
        )

        # MEMORY SAFE TFIDF
        tfidf = TfidfVectorizer(
            max_features=2000,   # reduced from 3000
            stop_words='english'
        )

        X = tfidf.fit_transform(df['synopsis'])

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df['genre'])

        # LIGHTER MODEL
        model = OneVsRestClassifier(
            LogisticRegression(max_iter=300)
        )

        model.fit(X, y)

    return tfidf, mlb, model


def predict_genres(text):
    text = text.lower()

    genres = {
        "action": ["fight", "battle", "war", "power"],
        "romance": ["love", "relationship", "couple"],
        "comedy": ["funny", "humor", "laugh"],
        "fantasy": ["magic", "kingdom", "dragon"],
        "sci-fi": ["space", "future", "technology"]
    }

    results = []

    for genre, keywords in genres.items():
        score = sum(1 for word in keywords if word in text)

        if score > 0:
            results.append((genre, score))

    return sorted(results, key=lambda x: x[1], reverse=True)[:5]