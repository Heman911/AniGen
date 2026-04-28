import pandas as pd

# Load datasets
df1 = pd.read_csv("data/anime_cleaned.csv")
df2 = pd.read_csv("data/anime_with_synopsis.csv")  # <-- your second file name

# Normalize column names
df1.columns = df1.columns.str.lower()
df2.columns = df2.columns.str.lower()

# Fix typo
df2 = df2.rename(columns={"sypnopsis": "synopsis"})

# Rename for consistency
df1 = df1.rename(columns={"name": "title"})
df2 = df2.rename(columns={"name": "title", "genres": "genre"})

# Merge datasets
df = pd.merge(df1, df2[['title', 'synopsis', 'genre']], on="title", how="inner")

# Save merged file
df.to_csv("data/anime_full.csv", index=False)

print("anime_full.csv created")