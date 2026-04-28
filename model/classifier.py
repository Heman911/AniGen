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


# Lightweight keyword-based classifier (Railway safe)

def predict_genres(text):
    if not text or not text.strip():
        return []

    text = text.lower()

    genre_keywords = {
        "action": ["fight", "battle", "war", "power", "weapon", "attack", "soldier"],
        "romance": ["love", "romance", "relationship", "couple", "crush", "kiss"],
        "comedy": ["funny", "humor", "laugh", "joke", "hilarious"],
        "fantasy": ["magic", "kingdom", "dragon", "demon", "fantasy", "spell"],
        "sci-fi": ["space", "future", "technology", "robot", "alien", "cyber"],
        "horror": ["ghost", "death", "kill", "blood", "horror", "dark"],
        "adventure": ["journey", "quest", "explore", "travel", "treasure"],
        "drama": ["life", "struggle", "emotion", "family", "past"],
        "slice of life": ["school", "daily", "life", "friends", "routine"],
        "sports": ["team", "match", "tournament", "goal", "competition"]
    }

    results = []

    for genre, keywords in genre_keywords.items():
        score = 0

        for word in keywords:
            if word in text:
                score += 1

        if score > 0:
            # convert to "confidence" (fake ML feel)
            confidence = round(min(0.95, 0.3 + score * 0.15), 2)
            results.append((genre, confidence))

    return sorted(results, key=lambda x: x[1], reverse=True)[:5]