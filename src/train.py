import pandas as pd
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import build_dataset

def laplace_smoothing(numerator, denominator, a=0.1):
    """avoid zero probability by applying Laplace smoothing"""
    return (numerator + a) / (denominator + 2 * a)

def predict_movie_rating(target_user_id, target_movie_id, df_train):

    # 1. Take the history of the target user
    target_user_history = df_train[df_train['user_id'] == target_user_id][['movie_id', 'rating']].copy()
    target_user_history['target_user_liked'] = (target_user_history['rating'] >= 4.0).astype(int)
    
    
    if target_user_history.empty:
        return 0


    other_users_df = df_train[df_train['user_id'] != target_user_id]
    target_movie_raters = other_users_df[other_users_df['movie_id'] == target_movie_id][['user_id', 'rating']].copy()
    

    if target_movie_raters.empty:
        return 0
        
    target_movie_raters.rename(columns={'rating': 'rating_on_target_movie'}, inplace=True)
    target_movie_raters['liked_target_movie'] = (target_movie_raters['rating_on_target_movie'] >= 4.0).astype(int)

    # Others history based on users who rated the target movie and also have history with target user
    relevant_history = other_users_df[
        (other_users_df['user_id'].isin(target_movie_raters['user_id'])) &
        (other_users_df['movie_id'].isin(target_user_history['movie_id']))
    ][['user_id', 'movie_id', 'rating']].copy()
    
    relevant_history['other_user_liked'] = (relevant_history['rating'] >= 4.0).astype(int)

    # merge relate to get same_preference

    merged_df = pd.merge(relevant_history, target_user_history[['movie_id', 'target_user_liked']], on='movie_id')

    merged_df['same_preference'] = (merged_df['other_user_liked'] == merged_df['target_user_liked'])
    same_pref_df = merged_df[merged_df['same_preference']]


    final_df = pd.merge(same_pref_df, target_movie_raters[['user_id', 'liked_target_movie']], on='user_id')

    
    if final_df.empty:
        return 0 

    stats = final_df.groupby('movie_id').agg(
        same_rating_time=('user_id', 'count'),         
        high_score_time=('liked_target_movie', 'sum')   
    ).reset_index()

    stats['low_score_time'] = stats['same_rating_time'] - stats['high_score_time']

    high_score_rating_PR = 1.0
    low_score_rating_PR = 1.0

    # Iteration through valid movie 
    for _, row in stats.iterrows():
        same_t = row['same_rating_time']
        high_t = row['high_score_time']
        low_t = row['low_score_time']
        
    
        high_score_rating_PR *= laplace_smoothing(high_t, same_t)
        low_score_rating_PR *= laplace_smoothing(low_t, same_t)

    return 1 if high_score_rating_PR >= low_score_rating_PR else 0


if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = build_dataset()
    
  
    df['actual_label'] = df['rating'].apply(lambda x: 1 if float(x) >= 4.0 else 0)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    
   
    unique_test_users = df_test['user_id'].unique()
    sampled_users = random.sample(list(unique_test_users), min(5, len(unique_test_users)))
    
    # Choose only the test samples of the sampled users
    df_test_sampled = df_test[df_test['user_id'].isin(sampled_users)]
    
    y_true = []
    y_pred = []
    
    print(f"Testing on {len(sampled_users)} user ")
    
   
    for index, row in df_test_sampled.iterrows():
        target_user = row['user_id']
        target_movie = row['movie_id']
        actual = row['actual_label']

        prediction = predict_movie_rating(target_user, target_movie, df_train)
        
        y_true.append(actual)
        y_pred.append(prediction)
        
   
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\n=== EVALUATION ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
   
    print("\nSaving model to 'naivebayes_model.pkl'...")
    with open('naivebayes_model.pkl', 'wb') as file:
        pickle.dump(df_train, file)
        
    print("Model saved!")