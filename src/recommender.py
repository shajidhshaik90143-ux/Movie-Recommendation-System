from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from .config import MODEL_PATH, MODEL_DIR

@dataclass
class MovieRecommender:
    movies: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def train(cls, movies: pd.DataFrame):
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1, max_features=12000, sublinear_tf=True)
        matrix = vectorizer.fit_transform(movies["content_text"])
        return cls(movies.reset_index(drop=True), vectorizer, matrix)

    def save(self, path=MODEL_PATH):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path=MODEL_PATH):
        return joblib.load(path)

    def _index(self, movie_id):
        hits = self.movies.index[self.movies.movie_id.astype(str) == str(movie_id)].tolist()
        return hits[0] if hits else None

    def similar(self, movie_id, n=10, diversity=0.15):
        idx = self._index(movie_id)
        if idx is None:
            return pd.DataFrame()
        sims = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        sims[idx] = -1
        candidates = np.argsort(sims)[::-1][:max(n*5, 20)]
        selected = []
        used_genres = set()
        for j in candidates:
            row = self.movies.iloc[j]
            genres = set(g.strip() for g in str(row.genres).split("|") if g.strip())
            novelty = 1.0 if not (genres & used_genres) else 1.0 - diversity
            score = sims[j] * (0.85 + 0.15 * novelty) + 0.08 * float(row.popularity_score) / 5.0
            selected.append((j, score))
            used_genres.update(genres)
            if len(selected) >= n:
                break
        selected.sort(key=lambda x: x[1], reverse=True)
        result = self.movies.iloc[[j for j,_ in selected]].copy()
        result["similarity"] = [float(s) for _,s in selected]
        return result

    def personalized(self, ratings: pd.DataFrame, user_id, n=10):
        user = ratings[ratings.user_id.astype(str) == str(user_id)].copy()
        if user.empty:
            return pd.DataFrame()
        liked = user[user.rating >= 4.0]
        if liked.empty:
            liked = user.sort_values("rating", ascending=False).head(3)
        indices = [self._index(mid) for mid in liked.movie_id]
        indices = [i for i in indices if i is not None]
        if not indices:
            return pd.DataFrame()
        weights = liked[liked.movie_id.astype(str).isin(self.movies.iloc[indices].movie_id.astype(str))]["rating"].to_numpy()
        if len(weights) != len(indices):
            weights = np.ones(len(indices))
        profile = self.matrix[indices].multiply(weights.reshape(-1, 1)).sum(axis=0)
        # scipy sparse sum(axis=0) returns a NumPy matrix; modern scikit-learn rejects np.matrix.
        profile = np.asarray(profile, dtype=np.float64)
        feature_matrix = self.matrix.tocsr()
        sims = cosine_similarity(profile, feature_matrix).ravel()
        seen = set(user.movie_id.astype(str))
        order = np.argsort(sims)[::-1]
        chosen = [i for i in order if str(self.movies.iloc[i].movie_id) not in seen][:n]
        result = self.movies.iloc[chosen].copy()
        result["personal_score"] = sims[chosen]
        return result

    def explain(self, source_id, target_id):
        a = self._index(source_id); b = self._index(target_id)
        if a is None or b is None: return "Similarity explanation unavailable."
        source = self.movies.iloc[a]; target = self.movies.iloc[b]
        sg = {x.strip().lower() for x in str(source.genres).split("|")}
        tg = {x.strip().lower() for x in str(target.genres).split("|")}
        common = sorted(x.title() for x in sg & tg if x)
        reasons = []
        if common: reasons.append("shared genres: " + ", ".join(common[:3]))
        if source.director == target.director and source.director: reasons.append(f"same director: {source.director}")
        shared_cast = sorted(set(str(source.cast).split(",")) & set(str(target.cast).split(",")))
        if shared_cast: reasons.append("shared cast: " + ", ".join(x.strip() for x in shared_cast[:2]))
        return " • ".join(reasons) if reasons else "similar themes, keywords, and story metadata"
