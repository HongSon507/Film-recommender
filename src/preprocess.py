from data_loader import (
    load_movies,
    load_ratings,
    load_users
)
def build_dataset():
    ratings = load_ratings()
    movies = load_movies()
    users = load_users()
    df = ratings.merge(users, on="user_id")
    df = df.merge(movies, on="movie_id")
    user_counts = df['user_id'].value_counts()
    movie_counts = df['movie_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= 20].index)]
    df = df[df['movie_id'].isin(movie_counts[movie_counts >= 20].index)]
    return df.reset_index(drop=True)

df = build_dataset()
print(df.columns.tolist())