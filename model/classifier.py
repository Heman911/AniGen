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

        # CLEAN DATA
        df.columns = df.columns.str.lower()
        df['synopsis'] = df['synopsis'].fillna("")

        # FIX GENRE COLUMN
        if 'genre_x' in df.columns:
            df['genre'] = df['genre_x']
        elif 'genre_y' in df.columns:
            df['genre'] = df['genre_y']
        else:
            raise Exception("No genre column found")

        df['genre'] = df['genre'].fillna("")
        df['genre'] = df['genre'].apply(
            lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
        )

        # TF-IDF
        tfidf = TfidfVectorizer(
            max_features=3000,   # reduced for memory
            stop_words='english'
        )

        X = tfidf.fit_transform(df['synopsis'])

        # LABELS
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(df['genre'])

        # MODEL
        model = OneVsRestClassifier(
            LogisticRegression(max_iter=500)  # lighter
        )
        model.fit(X, y)

    return tfidf, mlb, model


# PREDICTION FUNCTION
def predict_genres(text):
    if not text or not text.strip():
        return []

    tfidf, mlb, model = get_model()

    vec = tfidf.transform([text])
    probs = model.predict_proba(vec)

    results = []

    for i, genre in enumerate(mlb.classes_):
        score = probs[0][i]

        if score > 0.25:
            results.append((genre, round(score, 2)))

    return sorted(results, key=lambda x: x[1], reverse=True)[:5]