import pandas as pd
import pickle


def laplace_smoothing(numerator, denominator, a=0.1):
    """avoid zero probability by applying Laplace smoothing"""
    return (numerator + a) / (denominator + 2 * a)

def get_movie_like_probability(target_user_id, target_movie_id, df_train):
    target_user_history = df_train[df_train['user_id'] == target_user_id][['movie_id', 'rating']].copy()
    target_user_history['target_user_liked'] = (target_user_history['rating'] >= 4.0).astype(int)
    
    if target_user_history.empty:
        return 0.5  

    other_users_df = df_train[df_train['user_id'] != target_user_id]
    target_movie_raters = other_users_df[other_users_df['movie_id'] == target_movie_id][['user_id', 'rating']].copy()
    
    if target_movie_raters.empty:
        return 0.5

    target_movie_raters.rename(columns={'rating': 'rating_on_target_movie'}, inplace=True)
    target_movie_raters['liked_target_movie'] = (target_movie_raters['rating_on_target_movie'] >= 4.0).astype(int)

    relevant_history = other_users_df[
        (other_users_df['user_id'].isin(target_movie_raters['user_id'])) &
        (other_users_df['movie_id'].isin(target_user_history['movie_id']))
    ][['user_id', 'movie_id', 'rating']].copy()
    
    relevant_history['other_user_liked'] = (relevant_history['rating'] >= 4.0).astype(int)

    merged_df = pd.merge(relevant_history, target_user_history[['movie_id', 'target_user_liked']], on='movie_id')
    merged_df['same_preference'] = (merged_df['other_user_liked'] == merged_df['target_user_liked'])
    same_pref_df = merged_df[merged_df['same_preference']]

    final_df = pd.merge(same_pref_df, target_movie_raters[['user_id', 'liked_target_movie']], on='user_id')
    
    if final_df.empty:
        return 0.5 

    stats = final_df.groupby('movie_id').agg(
        same_rating_time=('user_id', 'count'),         
        high_score_time=('liked_target_movie', 'sum')   
    ).reset_index()
    stats['low_score_time'] = stats['same_rating_time'] - stats['high_score_time']

    high_score_rating_PR = 1.0
    low_score_rating_PR = 1.0

    for _, row in stats.iterrows():
        same_t = row['same_rating_time']
        high_t = row['high_score_time']
        low_t = row['low_score_time']
        
        high_score_rating_PR *= laplace_smoothing(high_t, same_t)
        low_score_rating_PR *= laplace_smoothing(low_t, same_t)

    # Tính xác suất Thích (PR >= 0.5 tương đương với Thích trong code cũ của bạn)
    sum_pr = high_score_rating_PR + low_score_rating_PR
    return high_score_rating_PR / sum_pr if sum_pr > 0 else 0.5


def recommend_top_n_movies(target_user_id, df_train, n=10):
  
    user_rated_movies = df_train[df_train['user_id'] == target_user_id]['movie_id'].unique()
    all_movies = df_train['movie_id'].unique()
    

    unseen_movies = [m for m in all_movies if m not in user_rated_movies]
    
    if not unseen_movies:
        print("This user did watch all movies!")
        return []

    print(f"-> Find {len(unseen_movies)} movies that user {target_user_id} has not rated yet. Calculating probabilities...")

# unseen movies 
    recommendations = []
    
    for movie_id in unseen_movies:
        prob = get_movie_like_probability(target_user_id, movie_id, df_train)
        
        if prob >= 0.5:
            recommendations.append((movie_id, prob))

    # arrangement by probability
    recommendations.sort(key=lambda x: x[1], reverse=True)

    return recommendations[:n]


if __name__ == "__main__":
    print("Finding model ...")
    with open('naivebayes_model.pkl', 'rb') as f:
        df_train = pickle.load(f)
   


   
    while True:
        print("\n" + "="*50)

        user_input = input("Press 'q' to quit or enter a User ID to get movie recommendations: ")
        
        if user_input.lower() == 'q':
            print("Goodbye!")
            break
            
        try:
    
            target_user_id = int(user_input)
            

            if target_user_id not in df_train['user_id'].values:
                print(f"ERROR: User ID '{target_user_id}' does not exist in the dataset. Please enter a different ID.")
                continue
                
            print(f"\n=== FINDING MOVIE RECOMMENDATIONS FOR USER ID: {target_user_id} ===")
            
  
            top_movies = recommend_top_n_movies(target_user_id, df_train, n=10)

     
            if top_movies:
                print(f"\n TOP {len(top_movies)} FILM RECOMMENDATIONS FOR USER {target_user_id}:")
                for i, (movie_id, prob) in enumerate(top_movies, 1):
                  
                    movie_row = df_train[df_train['movie_id'] == movie_id].iloc[0]
                    title = movie_row['title'] if 'title' in df_train.columns else f"Movie ID: {movie_id}"
                    
                    print(f"  {i}. {title} (Probability of liking: {prob*100:.2f}%)")
            else:
                print("\nNo movies found that meet the criteria for recommendation.")
                
        except ValueError:
            print("ERROR: User ID must be a valid integer!")