# utils.py
import pandas as pd

def laplace_smoothing(numerator, denominator, a=1):
    return (numerator + a) / (denominator + 2 * a)

def predict_movie_rating_with_prob(target_user_id, target_movie_id, df_train):
    target_user_history = df_train[df_train['user_id'] == target_user_id][['movie_id', 'rating']].copy()
    target_user_history['target_user_liked'] = (target_user_history['rating'] >= 4.0).astype(int)
    if target_user_history.empty: return 0, 0.5

    other_users_df = df_train[df_train['user_id'] != target_user_id]
    target_movie_raters = other_users_df[other_users_df['movie_id'] == target_movie_id][['user_id', 'rating']].copy()
    if target_movie_raters.empty: return 0, 0.5
        
    target_movie_raters.rename(columns={'rating': 'rating_on_target_movie'}, inplace=True)
    target_movie_raters['liked_target_movie'] = (target_movie_raters['rating_on_target_movie'] >= 4.0).astype(int)

    relevant_history = other_users_df[
        (other_users_df['user_id'].isin(target_movie_raters['user_id'])) &
        (other_users_df['movie_id'].isin(target_user_history['movie_id']))
    ][['user_id', 'movie_id', 'rating']].copy()
    if relevant_history.empty: return 0, 0.5
        
    relevant_history['other_user_liked'] = (relevant_history['rating'] >= 4.0).astype(int)
    merged_df = pd.merge(relevant_history, target_user_history[['movie_id', 'target_user_liked']], on='movie_id')
    merged_df['same_preference'] = (merged_df['other_user_liked'] == merged_df['target_user_liked'])
    final_df = pd.merge(merged_df[merged_df['same_preference']], target_movie_raters[['user_id', 'liked_target_movie']], on='user_id')
    if final_df.empty: return 0, 0.5

    stats = final_df.groupby('movie_id').agg(same_rating_time=('user_id', 'count'), high_score_time=('liked_target_movie', 'sum')).reset_index()
    stats['low_score_time'] = stats['same_rating_time'] - stats['high_score_time']

    high_score_rating_PR, low_score_rating_PR = 1.0, 1.0
    for _, row in stats.iterrows():
        high_score_rating_PR *= laplace_smoothing(row['high_score_time'], row['same_rating_time'])
        low_score_rating_PR *= laplace_smoothing(row['low_score_time'], row['same_rating_time'])

    sum_pr = high_score_rating_PR + low_score_rating_PR
    prob_like = high_score_rating_PR / sum_pr if sum_pr > 0 else 0.5
    return (1 if high_score_rating_PR >= low_score_rating_PR else 0), prob_like