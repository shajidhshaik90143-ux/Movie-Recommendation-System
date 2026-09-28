from __future__ import annotations
import pandas as pd
from .config import MOVIES_PATH, RATINGS_PATH

REQUIRED_MOVIE_COLUMNS = ["movie_id", "title", "year", "genres", "overview", "keywords", "director", "cast", "language"]

def load_movies(path=MOVIES_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_MOVIE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"movies.csv is missing columns: {missing}")
    # Keep movie_id in one canonical type across movies.csv and ratings.csv.
    # CSV files often infer IDs differently (e.g. object vs int64), which
    # causes pandas.merge() to fail with an object/int64 mismatch.
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    if df["movie_id"].isna().any():
        raise ValueError("movies.csv contains invalid movie_id values")
    df["movie_id"] = df["movie_id"].astype("int64")
    for c in REQUIRED_MOVIE_COLUMNS:
        if c != "movie_id":
            df[c] = df[c].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    return df

def load_ratings(path=RATINGS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["user_id", "movie_id", "rating"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"ratings.csv is missing columns: {missing}")
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
    df["movie_id"] = pd.to_numeric(df["movie_id"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
    df["user_id"] = df["user_id"].astype("int64")
    df["movie_id"] = df["movie_id"].astype("int64")
    return df

def prepare_movies(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    stats = ratings.groupby("movie_id")["rating"].agg(["count", "mean"]).rename(columns={"count":"rating_count", "mean":"avg_rating"})
    out = movies.merge(stats, on="movie_id", how="left")
    out["rating_count"] = out["rating_count"].fillna(0)
    out["avg_rating"] = out["avg_rating"].fillna(0)
    # Bayesian-style popularity score; prevents a movie with one 5-star rating from dominating.
    C = out.loc[out.rating_count > 0, "avg_rating"].mean() if (out.rating_count > 0).any() else 3.5
    m = max(float(out.rating_count.quantile(0.60)), 5.0)
    out["popularity_score"] = ((out.rating_count / (out.rating_count + m)) * out.avg_rating + (m / (out.rating_count + m)) * C)
    out["content_text"] = (
        out["genres"] + " " + out["genres"] + " " + out["keywords"] + " " +
        out["director"] + " " + out["cast"] + " " + out["overview"] + " " + out["language"]
    ).str.lower()
    return out
