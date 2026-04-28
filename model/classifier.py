import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# LOAD DATA
df = pd.read_csv("data/anime_full.csv")
df.columns = df.columns.str.lower()

# CLEAN DATA
# Fix synopsis
df['synopsis'] = df['synopsis'].fillna("")

# FIX MERGED GENRE COLUMNS
if 'genre_x' in df.columns:
    df['genre'] = df['genre_x']
elif 'genre_y' in df.columns:
    df['genre'] = df['genre_y']
else:
    raise Exception("❌ No genre column found (genre_x / genre_y missing)")

# Clean genres
df['genre'] = df['genre'].fillna("")
df['genre'] = df['genre'].apply(
    lambda x: [g.strip() for g in str(x).split(',') if g.strip()]
)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

X = tfidf.fit_transform(df['synopsis'])

# MULTI-LABEL ENCODING
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['genre'])

# TRAIN MODEL
model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model.fit(X, y)

# PREDICTION FUNCTION
def predict_genres(text):
    if not text or not text.strip():
        return []

    vec = tfidf.transform([text])
    probs = model.predict_proba(vec)

    results = []

    for i, genre in enumerate(mlb.classes_):
        score = probs[0][i]

        if score > 0.25:
            results.append((genre, round(score, 2)))

    return sorted(results, key=lambda x: x[1], reverse=True)[:5]