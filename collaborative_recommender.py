"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
           Option 3: Collaborative Filtering Movie Recommender System
========================================================================================
Description:
    An end-to-end Item-Based Collaborative Filtering Recommender System.
    Provides intelligent search-based recommendations, dataset analytics, and 
    comprehensive model evaluation (RMSE, MSE, MAE, Precision, Recall, F1-Score).

Key Capabilities:
    1. Fast Item-Based Collaborative Filtering (Pearson Correlation Matrix).
    2. Smart Title Search Engine (Handles partial names, typos, article reordering).
    3. Comprehensive Evaluation (80/20 Train-Test split for RMSE, MSE, MAE & Precision/Recall/F1).
    4. Dataset Explorer & Sparsity Analysis.
    5. Clean Interactive Console User Interface.
========================================================================================
"""

import os
import re
import difflib
import warnings
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress runtime warnings from sparse correlation calculations
warnings.filterwarnings('ignore')


# ======================================================================================
# 1. DATA LOADING MODULE
# ======================================================================================

def load_dataset(dataset_file='merged_movies_ratings.csv'):
    """
    Loads the merged MovieLens dataset ('merged_movies_ratings.csv').
    """
    try:
        if not os.path.exists(dataset_file):
            if os.path.exists('merged_movies_ratings.csv'):
                dataset_file = 'merged_movies_ratings.csv'
            else:
                print(f"[!] Error: Dataset file '{dataset_file}' not found.")
                print("    Please ensure 'merged_movies_ratings.csv' exists in the current directory.")
                return None
            
        print(f"[+] Loading dataset from '{dataset_file}'...")
        data = pd.read_csv(dataset_file)
        if 'tags' in data.columns:
            data['tags'] = data['tags'].fillna('')
            
        print(f"[+] Successfully loaded {len(data):,} ratings across {data['movieId'].nunique():,} unique movies.\n")
        return data
        
    except Exception as e:
        print(f"[!] Error loading dataset: {e}")
        return None


def build_recommender_matrix(data):
    """
    Builds the User-Item rating matrix (Pivot Table) and computes aggregate movie statistics.
    """
    print("[*] Building User-Item Interaction Matrix & Movie Statistics...")
    
    # Calculate movie-level statistics (including genres, tags, links)
    movie_stats = data.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        num_of_ratings=('rating', 'count'),
        genres=('genres', 'first'),
        tags=('tags', 'first') if 'tags' in data.columns else ('genres', lambda x: ''),
        movieId=('movieId', 'first')
    ).reset_index()
    
    movie_stats['tags'] = movie_stats['tags'].fillna('')
    
    # Create the User-Item matrix (rows = userId, columns = title)
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')
    
    num_users, num_movies = user_movie_matrix.shape
    print(f"[+] User-Item Matrix ready: {num_users} users x {num_movies} movies.\n")
    return user_movie_matrix, movie_stats


# ======================================================================================
# 2. INTELLIGENT ALL-SEARCH & QUERY MATCHING MODULE
# ======================================================================================

def normalize_title_query(query):
    """
    Generates variations for queries with leading articles.
    Example: 'The Matrix' -> ['The Matrix', 'Matrix, The', 'Matrix']
    """
    query_clean = query.strip()
    variants = [query_clean]
    for article in ['The ', 'A ', 'An ']:
        if query_clean.lower().startswith(article.lower()):
            variants.append(query_clean[len(article):].strip() + ', ' + article.strip())
            variants.append(query_clean[len(article):].strip())
    return variants


def search_movies(query, titles_list, movie_stats, max_results=5):
    """
    Multi-attribute all-search engine:
    1. Exact case-insensitive match on Title (with or without release year).
    2. Combined candidate scoring across Titles, Tags/Keywords, and Genres.
    3. Fuzzy string similarity fallback.
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    query_variants = normalize_title_query(query_clean)
    stats_map = dict(zip(movie_stats['title'], movie_stats['num_of_ratings']))
    
    # 1. Exact match on title variants
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if title.lower() == var_lower:
                return [title]
            clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
            if clean_title == var_lower:
                return [title]
                
    # 2. Gather candidates from Title, Tags, and Genres
    scored_candidates = {}
    
    # Title substring matches (highest weight)
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if var_lower in title.lower():
                pop = stats_map.get(title, 0)
                scored_candidates[title] = max(scored_candidates.get(title, 0), 1000 + pop)
                
    # Tag matches (weight: 500)
    if 'tags' in movie_stats.columns:
        tag_matches = movie_stats[movie_stats['tags'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in tag_matches.iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 500 + pop)
            
    # Genre matches (weight: 100)
    genre_matches = movie_stats[movie_stats['genres'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in genre_matches.iterrows():
        t = row['title']
        pop = row['num_of_ratings']
        scored_candidates[t] = max(scored_candidates.get(t, 0), 100 + pop)
        
    if scored_candidates:
        sorted_candidates = sorted(scored_candidates.keys(), key=lambda t: scored_candidates[t], reverse=True)
        return sorted_candidates[:max_results]
        
    # 3. Fuzzy similarity fallback
    fuzzy_matches = difflib.get_close_matches(query_clean, titles_list, n=max_results, cutoff=0.4)
    return fuzzy_matches


# ======================================================================================
# 3. ITEM-BASED COLLABORATIVE FILTERING RECOMMENDER ENGINE
# ======================================================================================

def get_collaborative_recommendations(movie_title, user_movie_matrix, movie_stats, min_ratings=50, min_overlap=15, top_n=10):
    """
    Generates movie recommendations using Item-Based Collaborative Filtering (Pearson Correlation).
    
    Parameters:
        movie_title (str): The exact target movie title.
        user_movie_matrix (pd.DataFrame): The User-Item rating matrix.
        movie_stats (pd.DataFrame): Summary statistics per movie.
        min_ratings (int): Minimum total ratings a candidate movie must have.
        min_overlap (int): Minimum co-rated users required between target and candidate.
        top_n (int): Number of recommendations to return.
    """
    if movie_title not in user_movie_matrix.columns:
        return None
        
    target_ratings = user_movie_matrix[movie_title]
    target_non_null = target_ratings.notna()
    
    # Filter candidates by minimum total ratings for speed and statistical significance
    popular_titles = movie_stats[movie_stats['num_of_ratings'] >= min_ratings]['title']
    if movie_title not in popular_titles.values:
        popular_titles = pd.concat([popular_titles, pd.Series([movie_title])])
        
    candidate_matrix = user_movie_matrix[popular_titles]
    
    correlations = {}
    for col in candidate_matrix.columns:
        if col == movie_title:
            continue
            
        col_ratings = candidate_matrix[col]
        common_users = target_non_null & col_ratings.notna()
        overlap_count = common_users.sum()
        
        if overlap_count >= min_overlap:
            x = target_ratings[common_users]
            y = col_ratings[common_users]
            if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                corr = np.corrcoef(x, y)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = (corr, overlap_count)
                    
    if not correlations:
        return pd.DataFrame()
        
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation', 'Co-rated Users'])
    corr_df.index.name = 'title'
    
    results = corr_df.reset_index().merge(movie_stats, on='title')
    results['avg_rating'] = results['avg_rating'].round(2)
    results['Correlation'] = results['Correlation'].round(4)
    
    # Sort primarily by correlation, secondarily by total rating count
    recommendations = results.sort_values(by=['Correlation', 'num_of_ratings'], ascending=[False, False])
    
    output_cols = ['title', 'Correlation', 'avg_rating', 'num_of_ratings', 'Co-rated Users', 'genres']
    return recommendations[output_cols].head(top_n).reset_index(drop=True)


# ======================================================================================
# 4. SYSTEM EVALUATION MODULE (RMSE, MSE, MAE, PRECISION, RECALL, F1)
# ======================================================================================

def evaluate_recommender_system(data, test_size=0.2, random_state=42, relevance_threshold=3.5):
    """
    Evaluates the recommender system using an 80/20 Train-Test split.
    Calculates:
      1. Rating Prediction Metrics: Root Mean Squared Error (RMSE), Mean Squared Error (MSE), Mean Absolute Error (MAE).
      2. Recommendation Quality Metrics: Precision, Recall, F1-Score, and Classification Accuracy.
    """
    print("\n" + "="*75)
    print("      [EVALUATION] RECOMMENDER SYSTEM ACCURACY (80/20 Train-Test Split)")
    print("="*75)
    
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"[*] Training Ratings : {len(train_df):,} ratings (80%)")
    print(f"[*] Testing Ratings  : {len(test_df):,} ratings (20%)")
    print(f"[*] Relevance Cutoff : Rating >= {relevance_threshold:.1f} stars\n")
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Global Mean Baseline
    pred_global = [global_mean] * len(test_df)
    mse_global = mean_squared_error(test_df['rating'], pred_global)
    rmse_global = sqrt(mse_global)
    mae_global = mean_absolute_error(test_df['rating'], pred_global)
    
    # 2. Movie Average Baseline
    pred_movie = [movie_means.get(m, global_mean) for m in test_df['movieId']]
    mse_movie = mean_squared_error(test_df['rating'], pred_movie)
    rmse_movie = sqrt(mse_movie)
    mae_movie = mean_absolute_error(test_df['rating'], pred_movie)
    
    # 3. User Average Baseline
    pred_user = [user_means.get(u, global_mean) for u in test_df['userId']]
    mse_user = mean_squared_error(test_df['rating'], pred_user)
    rmse_user = sqrt(mse_user)
    mae_user = mean_absolute_error(test_df['rating'], pred_user)
    
    # 4. User + Movie Bias (Collaborative Baseline)
    pred_combined = [
        np.clip(user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean, 0.5, 5.0)
        for u, m in zip(test_df['userId'], test_df['movieId'])
    ]
    mse_combined = mean_squared_error(test_df['rating'], pred_combined)
    rmse_combined = sqrt(mse_combined)
    mae_combined = mean_absolute_error(test_df['rating'], pred_combined)
    
    # Classification Metrics (Precision, Recall, F1)
    actual_binary = (test_df['rating'] >= relevance_threshold).astype(int)
    pred_binary = (np.array(pred_combined) >= relevance_threshold).astype(int)
    
    tp = ((pred_binary == 1) & (actual_binary == 1)).sum()
    fp = ((pred_binary == 1) & (actual_binary == 0)).sum()
    fn = ((pred_binary == 0) & (actual_binary == 1)).sum()
    tn = ((pred_binary == 0) & (actual_binary == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(actual_binary)
    
    # Rating Prediction Error Table
    eval_table = pd.DataFrame([
        {"Predictive Model / Baseline": "1. Global Mean Rating", "MSE": round(mse_global, 4), "RMSE": round(rmse_global, 4), "MAE": round(mae_global, 4)},
        {"Predictive Model / Baseline": "2. Movie Average Rating", "MSE": round(mse_movie, 4), "RMSE": round(rmse_movie, 4), "MAE": round(mae_movie, 4)},
        {"Predictive Model / Baseline": "3. User Average Rating", "MSE": round(mse_user, 4), "RMSE": round(rmse_user, 4), "MAE": round(mae_user, 4)},
        {"Predictive Model / Baseline": "4. User + Movie Bias (CF Baseline)", "MSE": round(mse_combined, 4), "RMSE": round(rmse_combined, 4), "MAE": round(mae_combined, 4)},
    ])
    
    print("--- [A] Rating Prediction Accuracy (Lower is Better) ---")
    print(eval_table.to_string(index=False))
    
    print("\n--- [B] Recommendation Classification Quality (Higher is Better) ---")
    metrics_table = pd.DataFrame([
        {"Metric": "Precision (Relevant Recommendations)", "Score": f"{precision:.4f} ({precision*100:.2f}%)"},
        {"Metric": "Recall (Discovered Relevant Movies)", "Score": f"{recall:.4f} ({recall*100:.2f}%)"},
        {"Metric": "F1-Score (Harmonic Mean)", "Score": f"{f1:.4f} ({f1*100:.2f}%)"},
        {"Metric": "Classification Accuracy", "Score": f"{accuracy:.4f} ({accuracy*100:.2f}%)"},
    ])
    print(metrics_table.to_string(index=False))
    print("="*75 + "\n")


# ======================================================================================
# 5. DATASET SUMMARY & ANALYTICS MODULE
# ======================================================================================

def display_dataset_summary(data):
    """
    Displays comprehensive statistics and dataset properties.
    """
    print("\n" + "="*75)
    print("              [ANALYTICS] DATASET SUMMARY & EXPLORATORY METRICS")
    print("="*75)
    
    num_ratings = len(data)
    num_users = data['userId'].nunique()
    num_movies = data['movieId'].nunique()
    total_possible = num_users * num_movies
    sparsity = (1.0 - (num_ratings / total_possible)) * 100
    
    print(f"Total User Ratings   : {num_ratings:,}")
    print(f"Unique Users         : {num_users:,}")
    print(f"Unique Movies        : {num_movies:,}")
    print(f"Rating Scale         : {data['rating'].min()} to {data['rating'].max()} stars")
    print(f"Average Rating       : {data['rating'].mean():.2f} stars")
    print(f"Rating Matrix Size   : {num_users} x {num_movies} ({total_possible:,} cells)")
    print(f"Matrix Sparsity      : {sparsity:.2f}% (Standard in Recommender Systems)")
    
    print("\nTop 5 Most Rated Movies:")
    top_rated = data.groupby('title').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).sort_values('num_ratings', ascending=False).head(5).reset_index()
    top_rated['avg_rating'] = top_rated['avg_rating'].round(2)
    print(top_rated.to_string(index=False))
    print("="*75 + "\n")


# ======================================================================================
# 6. INTERACTIVE CLI / MAIN APPLICATION LOOP
# ======================================================================================

def interactive_search_mode(user_movie_matrix, movie_stats):
    """
    Handles interactive user queries for movie recommendations.
    """
    titles_list = movie_stats['title'].tolist()
    
    while True:
        print("\n" + "-"*75)
        try:
            user_input = input("Enter movie name/keyword (or 'b' for main menu): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ('b', 'back', 'exit', 'q', 'quit'):
            break
            
        matches = search_movies(user_input, titles_list, movie_stats, max_results=5)
        
        if not matches:
            print(f"\n[!] No movies found matching '{user_input}'.")
            print("    Hint: Try typing a single keyword (e.g. 'Jedi', 'Matrix', 'Avatar', 'Batman').")
            continue
            
        # Movie Selection
        if len(matches) == 1:
            target_movie = matches[0]
        else:
            print(f"\nMultiple movies matched '{user_input}':")
            for idx, title in enumerate(matches, 1):
                cnt = movie_stats.loc[movie_stats['title'] == title, 'num_of_ratings'].values[0]
                avg = movie_stats.loc[movie_stats['title'] == title, 'avg_rating'].values[0]
                print(f"  [{idx}] {title} ({cnt} ratings, Avg: {avg:.2f}/5.0)")
                
            try:
                choice = input(f"Select a movie [1-{len(matches)}] (default 1): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                target_movie = matches[int(choice) - 1]
            else:
                target_movie = matches[0]
                
        print(f"\n[*] Generating collaborative recommendations for: '{target_movie}'...")
        recs = get_collaborative_recommendations(target_movie, user_movie_matrix, movie_stats, min_ratings=50, top_n=10)
        
        if recs is None or recs.empty:
            print("[!] No recommendations found meeting the correlation threshold.")
        else:
            print("\n>>> Top Recommendations:")
            print(recs.to_string(index=False))


def main():
    print("="*75)
    print("     [RECOMMENDER SYSTEM] COLLABORATIVE FILTERING MOVIE RECOMMENDER")
    print("                 TARUMT - Artificial Intelligence Project")
    print("="*75 + "\n")
    
    # 1. Load Data
    data = load_dataset()
    if data is None:
        return
        
    # 2. Build User-Item Interaction Matrix
    user_movie_matrix, movie_stats = build_recommender_matrix(data)
    
    # 3. Main Console Menu Loop
    while True:
        print("\n" + "="*50)
        print("                 MAIN MENU")
        print("="*50)
        print("  [1] Search Movie & Get Recommendations")
        print("  [2] Run Recommender System Evaluation (RMSE/MAE/F1)")
        print("  [3] View Dataset Summary & Statistics")
        print("  [4] Exit Application")
        print("="*50)
        
        try:
            choice = input("Enter your option [1-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Recommender System. Goodbye!")
            break
            
        if choice == '1':
            interactive_search_mode(user_movie_matrix, movie_stats)
        elif choice == '2':
            evaluate_recommender_system(data)
        elif choice == '3':
            display_dataset_summary(data)
        elif choice in ('4', 'exit', 'quit', 'q'):
            print("\nThank you for using the Recommender System. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter a number from 1 to 4.")


if __name__ == "__main__":
    main()
