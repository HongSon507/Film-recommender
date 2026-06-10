import pandas as pd
import pickle

def laplace_smoothing(numerator, denominator, a=0.1):
    return (numerator + a) / (denominator + 2 * a)

def get_movie_like_probability(target_user_id, target_movie_id, df_train):
    target_user_history = df_train[df_train['user_id'] == target_user_id][['movie_id', 'rating']].copy()
    target_user_history['target_user_liked'] = (target_user_history['rating'] >= 4.0).astype(int)
    if target_user_history.empty: return 0.5

    other_users_df = df_train[df_train['user_id'] != target_user_id]
    target_movie_raters = other_users_df[other_users_df['movie_id'] == target_movie_id][['user_id', 'rating']].copy()
    if target_movie_raters.empty: return 0.5

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
    if final_df.empty: return 0.5 

    stats = final_df.groupby('movie_id').agg(
        same_rating_time=('user_id', 'count'),         
        high_score_time=('liked_target_movie', 'sum')   
    ).reset_index()
    stats['low_score_time'] = stats['same_rating_time'] - stats['high_score_time']

    high_score_rating_PR, low_score_rating_PR = 1.0, 1.0
    for _, row in stats.iterrows():
        same_t = row['same_rating_time']
        high_score_rating_PR *= laplace_smoothing(row['high_score_time'], same_t)
        low_score_rating_PR *= laplace_smoothing(row['low_score_time'], same_t)

    sum_pr = high_score_rating_PR + low_score_rating_PR
    return high_score_rating_PR / sum_pr if sum_pr > 0 else 0.5

def recommend_top_n_movies(target_user_id, df_train, n=10):
    user_rated_movies = df_train[df_train['user_id'] == target_user_id]['movie_id'].unique()
    all_movies = df_train['movie_id'].unique()
    unseen_movies = [m for m in all_movies if m not in user_rated_movies]
    
    print(f"-> Processing {len(unseen_movies)} unseen movies. Please wait...")
    
    recommendations = []
    for movie_id in unseen_movies:
        prob = get_movie_like_probability(target_user_id, movie_id, df_train)
        if prob >= 0.53: 
            recommendations.append((movie_id, prob))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]


if __name__ == "__main__":
    print("Loading system data...")
    try:
        with open('naivebayes_model.pkl', 'rb') as f:
            df_train = pickle.load(f)
    except FileNotFoundError:
        print("ERROR: Model file not found. Please run 'train.py' first.")
        exit()

    while True:
        print("\n" + "="*50)
        print("MOVIE RECOMMENDER SYSTEM")
        print("1. Login with existing User ID")
        print("2. Create New User & Rate Movies")
        print("3. Quit (press 'q')")
        
        choice = input("Select an option (1, 2, or q): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye!")
            break
            
        elif choice == '1':
            user_input = input("Enter User ID: ")
            try:
                target_user_id = int(user_input)
                if target_user_id not in df_train['user_id'].values:
                    print("ERROR: User ID does not exist.")
                    continue
            except ValueError:
                print("ERROR: Invalid ID format.")
                continue
                
        elif choice == '2':
            print("\n--- NEW USER PREFERENCE SURVEY ---")
            print("Rate the following popular movies from 1 to 5 (Enter 0 if unseen).")
            
            popular_movie_ids = df_train['movie_id'].value_counts().head(5).index
            new_user_ratings = []
            target_user_id = 999999 
            
            for m_id in popular_movie_ids:
                title = df_train[df_train['movie_id'] == m_id]['title'].iloc[0] if 'title' in df_train.columns else f"Movie {m_id}"
                
                while True:
                    try:
                        rate = float(input(f"- '{title}': "))
                        if rate == 0 or (1.0 <= rate <= 5.0): break
                        print("  Please enter a number between 1 and 5 (or 0).")
                    except ValueError:
                        print("  Invalid input. Please enter a number.")
                
                if rate > 0:
                    new_user_ratings.append({
                        'user_id': target_user_id,
                        'movie_id': m_id,
                        'rating': rate,
                        'title': title if 'title' in df_train.columns else None
                    })
            
            if not new_user_ratings:
                print("No ratings provided. Cannot generate recommendations.")
                continue
                
            df_new_user = pd.DataFrame(new_user_ratings)
            df_train = pd.concat([df_train, df_new_user], ignore_index=True)
            print("\n-> User profile created successfully!")

        else:
            print("Invalid selection.")
            continue

        print(f"\n=== RECOMMENDATIONS FOR USER {target_user_id} ===")
        top_movies = recommend_top_n_movies(target_user_id, df_train, n=10)

        if top_movies:
            print(f"\n🏆 TOP {len(top_movies)} MOVIE MATCHES:")
            for i, (m_id, prob) in enumerate(top_movies, 1):
                title = df_train[df_train['movie_id'] == m_id]['title'].iloc[0] if 'title' in df_train.columns else f"Movie {m_id}"
                print(f"  {i}. {title} (Confidence: {prob*100:.2f}%)")
        else:
            print("\nNo suitable movies found for recommendation.")