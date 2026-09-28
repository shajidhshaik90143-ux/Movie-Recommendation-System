from src.data_loader import load_movies, load_ratings, prepare_movies
from src.recommender import MovieRecommender

if __name__ == "__main__":
    movies = load_movies()
    ratings = load_ratings()
    movies = prepare_movies(movies, ratings)
    model = MovieRecommender.train(movies)
    model.save()
    print(f"Training complete: {len(movies)} movies, {model.matrix.shape[1]} TF-IDF features")
    print("Saved: models/recommender.joblib")
