# 🎌 AniGen

AniGen is a **machine learning-based anime recommendation and genre classification web application** built using Flask. It allows users to discover anime based on similarity, filter by genre, and predict genres from descriptions using NLP techniques.

---

##  Features

### Anime Recommendation
- Enter an anime title
- Get similar anime using **content-based filtering**
- Uses **cosine similarity on genre vectors**

### Genre-Based Recommendation
- Search anime by genre (e.g., action, romance)
- Returns filtered anime list with details

### Genre Classification (NLP)
- Input an anime description
- Predict genres using:
  - **TF-IDF vectorization**
  - **Multi-label classification (Logistic Regression)**

### Smart Autocomplete
- Real-time suggestions while typing
- Uses **fuzzy matching (RapidFuzz)**

### Dynamic Posters
- Fetches anime images using **Jikan API**
- Caches results for faster loading

---

## Technologies Used

- **Backend:** Flask (Python)
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF (Text Vectorization)
- **Frontend:** HTML, CSS, JavaScript
- **API:** Jikan (MyAnimeList API)
- **Data Processing:** Pandas

---

## Machine Learning Components

### Recommendation System
- MultiLabelBinarizer for genre encoding
- Cosine similarity for finding similar anime

### Genre Classifier
- TF-IDF vectorizer on synopsis text
- One-vs-Rest Logistic Regression
- Multi-label genre prediction

### Search Optimization
- Fuzzy string matching for autocomplete

---

## Project Structure
AniGen/
│
├── app.py
├── model/
│ ├── recommender.py
│ └── classifier.py
│
├── data/
│ ├── anime_cleaned.csv
│ ├── anime_with_synopsis.csv
│ └── anime_full.csv
│
├── templates/
│ ├── index.html
│ ├── recommend.html
│ └── classify.html
│
├── static/
│ ├── forBG.mp4
│ ├── New.mp4
│ └── default.png
│
└── image_cache.json


## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/anigen.git
cd anigen

### 2. Install dependencies
pip install flask pandas scikit-learn rapidfuzz requests
### 3. Run the application
python app.py
### 4. Open in browser
http://127.0.0.1:5000/
 Dataset

The project uses:

Anime metadata dataset
Synopsis dataset
Merged into anime_full.csv

 Note:
Original image URLs were unreliable, so images are fetched dynamically using the Jikan API.

 UI Highlights
Animated video backgrounds
Glassmorphism input fields
Smooth page transitions
Responsive grid layout for recommendations
 Known Issues
Some genres may return limited results depending on dataset coverage
API image fetching may occasionally be slow
 Future Improvements
Dropdown genre selector
Anime rating & popularity sorting
Detailed anime info popup
Better UI animations (Netflix-style)
Model accuracy improvements
 License

This project is for educational purposes.

 Acknowledgements
MyAnimeList (via Jikan API)
Scikit-learn documentation
Open anime datasets