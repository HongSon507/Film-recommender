
import pandas as pd
def load_movies():
    movies = pd.read_csv(r"/home/sontran/projects/Film-recommender/data/ml-100k/u.item",    sep="|",
        encoding="latin-1",
        header=None
    )
    
    movie_columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "FilmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western"]
    movies.columns = movie_columns
    return movies

def load_ratings():
    ratings = pd.read_csv(r"/home/sontran/projects/Film-recommender/data/ml-100k/u.data", sep="\t", names = ["user_id", "movie_id", "rating", "timestamp"])
    ratings['labels'] = (ratings['rating'] >= 4).astype(int)
    return ratings
def load_users():
    users = pd.read_csv(r"/home/sontran/projects/Film-recommender/data/ml-100k/u.user", sep="|", names = ["user_id", "age", "gender", "occupation", "zip_code"])
    return users