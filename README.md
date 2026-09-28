# Movie Recommendation System

A strong, portfolio-ready movie recommendation web app built with Python and Streamlit.

## Features
- Hybrid recommendation engine combining content similarity and popularity.
- TF-IDF vectorization over genres, keywords, director, cast, overview, and tags.
- Cosine-similarity recommendations with configurable diversity.
- Personalized recommendations from a user's ratings profile.
- Trending/popular movies and genre exploration.
- Search by title, genre, year, rating, and language.
- Movie details, similar movies, and recommendation explanations.
- Interactive Plotly analytics dashboard.
- Offline sample dataset included; no API key required.
- Train/cache pipeline using Joblib.
- Clean package structure with `src` imports (avoids common `ModuleNotFoundError` issues).

## Project Structure
```text
Movie_Recommendation_System/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── run_windows.bat
├── run_linux.sh
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── models/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── recommender.py
│   └── analytics.py
├── assets/
└── .streamlit/
    └── config.toml
```

## Requirements
Recommended: Python 3.10–3.13.

## Windows Setup
```powershell
cd "Movie_Recommendation_System"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
streamlit run app.py
```

If `py -3.12` is unavailable, use your installed Python version. Python 3.14 may not yet have wheels for every scientific dependency on every machine.

## Simple Setup
```powershell
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Linux/macOS
```bash
chmod +x run_linux.sh
./run_linux.sh
```

## How the model works
1. Movie metadata is cleaned and combined into a `content_text` field.
2. TF-IDF converts the text into numerical vectors.
3. Cosine similarity finds movies with similar metadata.
4. Popularity is calculated from rating count and average rating using a Bayesian-style weighted score.
5. Hybrid ranking blends content similarity with popularity and a small genre-diversity penalty.
6. User-profile recommendations aggregate the content vectors of movies the user rated highly.

## Important
The included dataset is an original demo dataset designed to make the project runnable offline. For a real deployment, replace it with a properly licensed dataset such as MovieLens and retrain.
