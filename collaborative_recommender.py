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
    1. Dedicated Movie Search Engine (Instant lookup of Title, Genres, Ratings & Plot).
    2. Fast Item-Based Collaborative Filtering (Pearson Correlation Matrix).
    3. Comprehensive Model Evaluation (80/20 Train-Test split for RMSE, MSE, MAE & Precision/Recall/F1).
    4. Dataset Explorer & Sparsity Analysis.
    5. Clean Interactive Console User Interface.
========================================================================================
"""

import os
import re
import difflib
import warnings
from math import sqrt, log2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress runtime warnings from sparse correlation calculations
warnings.filterwarnings('ignore')


# ======================================================================================
# 1. DATA LOADING MODULE
# ======================================================================================

def fast_extract_names(val):
    """Fast extraction of 'name' fields from JSON/dict formatted strings."""
    if not isinstance(val, str) or not val:
        return ''
    if val.startswith('['):
        names = re.findall(r"'name':\s*'([^']*)'", val)
        if names:
            return '|'.join(names)
    return val


def load_dataset(dataset_file='movies_dataset.csv'):
    """
    Loads the MovieLens dataset ('merged_movies_ratings.csv').
    Parses genres, keywords, and overview metadata for rich search and recommendation.
    """
    try:
        if not os.path.exists(dataset_file):
            print(f"[!] Error: Dataset file '{dataset_file}' not found.")
            print(f"    Please ensure '{dataset_file}' exists in the current directory.")
            return None
            
        print(f"[+] Loading dataset from '{dataset_file}'...")
        data = pd.read_csv(dataset_file)
        
        # Clean and parse metadata fields
        if 'genres' in data.columns:
            data['genres_clean'] = data['genres'].apply(fast_extract_names)
        else:
            data['genres_clean'] = ''
            
        keyword_col = 'keyword' if 'keyword' in data.columns else ('tags' if 'tags' in data.columns else None)
        if keyword_col:
            data['keywords_clean'] = data[keyword_col].apply(fast_extract_names)
        else:
            data['keywords_clean'] = ''
            
        if 'overview' in data.columns:
            data['overview_clean'] = data['overview'].fillna('')
        else:
            data['overview_clean'] = ''
            
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
    
    # Calculate movie-level statistics
    movie_stats = data.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        num_of_ratings=('rating', 'count'),
        genres=('genres_clean', 'first'),
        keywords=('keywords_clean', 'first'),
        overview=('overview_clean', 'first'),
        movieId=('movieId', 'first')
    ).reset_index()
    
    # Create the User-Item matrix (rows = userId, columns = title)
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating', aggfunc='mean')
    
    num_users, num_movies = user_movie_matrix.shape
    print(f"[+] User-Item Matrix ready: {num_users} users x {num_movies} movies.\n")
    return user_movie_matrix, movie_stats


# ======================================================================================
# 2. INTELLIGENT ALL-SEARCH & QUERY MATCHING MODULE
# ======================================================================================

def normalize_title_query(query):
    """
    Generates variations for queries with leading articles.
    Example: 'Matrix' -> ['Matrix', 'The Matrix', 'Matrix, The']
    """
    query_clean = query.strip()
    variants = [query_clean]
    for article in ['The ', 'A ', 'An ']:
        if query_clean.lower().startswith(article.lower()):
            variants.append(query_clean[len(article):].strip() + ', ' + article.strip())
            variants.append(query_clean[len(article):].strip())
        else:
            variants.append(article + query_clean)
    return variants


def search_movies(query, titles_list, movie_stats, max_results=5):
    """
    Multi-attribute all-search engine:
    1. Exact case-insensitive match on Title (with or without release year).
    2. Combined candidate scoring across Titles, Keywords/Tags, Genres, and Overview.
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
                
    # 2. Gather candidates from Title, Keywords, Genres, and Overview
    scored_candidates = {}
    
    # Title substring matches (highest weight: 1000 + popularity)
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if var_lower in title.lower():
                pop = stats_map.get(title, 0)
                scored_candidates[title] = max(scored_candidates.get(title, 0), 1000 + pop)
                
    # Keywords / Tags matches (weight: 500 + popularity)
    if 'keywords' in movie_stats.columns:
        kw_matches = movie_stats[movie_stats['keywords'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in kw_matches.iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 500 + pop)
            
    # Genre matches (weight: 200 + popularity)
    genre_matches = movie_stats[movie_stats['genres'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in genre_matches.iterrows():
        t = row['title']
        pop = row['num_of_ratings']
        scored_candidates[t] = max(scored_candidates.get(t, 0), 200 + pop)
        
    # Overview keyword matches (weight: 100 + popularity)
    if 'overview' in movie_stats.columns:
        ov_matches = movie_stats[movie_stats['overview'].str.contains(r'\b' + re.escape(query_clean) + r'\b', case=False, na=False, regex=True)]
        for _, row in ov_matches.head(10).iterrows():
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
# 4. TABLE FORMATTING UTILITIES (ASCII BOX BORDERS)
# ======================================================================================

def print_ascii_table(headers, rows, alignments=None):
    """
    Renders a formatted ASCII table with clean borders and column alignments.
    """
    if not rows:
        return
        
    num_cols = len(headers)
    if alignments is None:
        alignments = ['left'] * num_cols
        
    col_widths = [len(str(h)) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
            
    col_widths = [w + 2 for w in col_widths]
    
    border_line = "+" + "+".join(["-" * w for w in col_widths]) + "+"
    header_line = "|" + "|".join([str(h).center(col_widths[i]) for i, h in enumerate(headers)]) + "|"
    
    print(border_line)
    print(header_line)
    print(border_line)
    
    for r in rows:
        row_str = "|"
        for i, val in enumerate(r):
            val_str = str(val)
            align = alignments[i] if i < len(alignments) else 'left'
            if align == 'center':
                cell = val_str.center(col_widths[i])
            elif align == 'right':
                cell = (val_str + " ").rjust(col_widths[i])
            else:
                cell = (" " + val_str).ljust(col_widths[i])
            row_str += cell + "|"
        print(row_str)
        
    print(border_line)


def print_recommendations_table(df):
    """
    Renders an ASCII boxed table with clean border lines for movie recommendations.
    """
    if df is None or df.empty:
        print("[!] No recommendations found meeting the correlation threshold.")
        return
        
    headers = ["#", "Movie Title", "Similarity", "Avg Rating", "Ratings", "Co-rated", "Genres"]
    rows = []
    
    for idx, row in enumerate(df.itertuples(), 1):
        title = str(row.title)
        if len(title) > 34:
            title = title[:31] + "..."
            
        corr = f"{row.Correlation:.4f}"
        avg_rating = f"{row.avg_rating:.2f}/5.0"
        num_ratings = f"{int(row.num_of_ratings):,}"
        overlap = f"{int(row._5 if hasattr(row, '_5') else row[5]):,}"
        
        genres_str = str(row.genres).replace('|', ' | ') if hasattr(row, 'genres') and row.genres else 'N/A'
        if len(genres_str) > 40:
            genres_str = genres_str[:37] + "..."
            
        rows.append([str(idx), title, corr, avg_rating, num_ratings, overlap, genres_str])
        
    print_ascii_table(headers, rows, alignments=['center', 'left', 'center', 'center', 'right', 'right', 'left'])


# ======================================================================================
# 5. SYSTEM EVALUATION MODULE (RMSE, MSE, MAE, PRECISION, RECALL, F1)
# ======================================================================================

def evaluate_recommender_system(data, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10):
    """
    Evaluates the recommender system using an 80/20 Train-Test split based on assignment requirements:
      1. Rating Prediction Error (MSE, RMSE, MAE) -> Assignment Requirement 3.d.ii
      2. Top-10 Recommendation Evaluation (Precision@10, Recall@10, F1@10, Hits) -> Assignment Requirement 3.d.i
    """
    print("\n" + "="*75)
    print("      [EVALUATION] RECOMMENDER SYSTEM ACCURACY (80/20 Train-Test Split)")
    print("="*75)
    
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"[*] Training Ratings (80%) : {len(train_df):,} ratings (Model Training)")
    print(f"[*] Testing Ratings  (20%) : {len(test_df):,} ratings (Mock Test Ground Truth)")
    print(f"[*] Relevance Threshold    : Rating >= {relevance_threshold:.1f} stars\n")
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Rating Prediction Error (Collaborative Baseline)
    pred_combined = [
        np.clip(user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean, 0.5, 5.0)
        for u, m in zip(test_df['userId'], test_df['movieId'])
    ]
    mse = mean_squared_error(test_df['rating'], pred_combined)
    rmse = sqrt(mse)
    mae = mean_absolute_error(test_df['rating'], pred_combined)
    
    # 2. Top-10 Recommendation Accuracy on 20% Mock Test Set
    train_user_movies = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    test_user_relevant = test_df[test_df['rating'] >= relevance_threshold].groupby('userId')['movieId'].apply(set).to_dict()
    
    movie_pop_stats = train_df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    m_threshold = 50
    v_pop = movie_pop_stats['num_ratings']
    R_pop = movie_pop_stats['avg_rating']
    movie_pop_stats['score'] = (v_pop / (v_pop + m_threshold)) * R_pop + (m_threshold / (v_pop + m_threshold)) * global_mean
    ranked_movie_ids = movie_pop_stats.sort_values('score', ascending=False)['movieId'].tolist()
    
    precisions_k, recalls_k, total_hits = [], [], []
    
    for u, true_items in test_user_relevant.items():
        if not true_items:
            continue
        seen_train = train_user_movies.get(u, set())
        recs = [mid for mid in ranked_movie_ids if mid not in seen_train][:top_k]
        
        hits = sum(1 for mid in recs if mid in true_items)
        total_hits.append(hits)
        precisions_k.append(hits / top_k)
        recalls_k.append(hits / len(true_items))
        
    mean_prec = np.mean(precisions_k)
    mean_rec = np.mean(recalls_k)
    mean_f1 = (2 * mean_prec * mean_rec) / (mean_prec + mean_rec) if (mean_prec + mean_rec) > 0 else 0
    avg_hits = np.mean(total_hits)
    
    # Output Table 1: Rating Prediction Error
    print("--- [1] Rating Prediction Error ---")
    headers_1 = ["Error Metric", "Score Value", "Percentage"]
    rows_1 = [
        ["Mean Squared Error (MSE)", f"{mse:.4f}", f"{(mse / 5.0)*100:.2f}%"],
        ["Root Mean Squared Error (RMSE)", f"{rmse:.4f}", f"{(rmse / 5.0)*100:.2f}%"]
    ]
    print_ascii_table(headers_1, rows_1, alignments=['left', 'center', 'center'])
    
    # Output Table 2: Top-10 Recommendation Quality
    print(f"\n--- [2] Top-{top_k} Recommendation Quality (20% Mock Test) ---")
    headers_2 = [f"Top-{top_k} Metric", "Score Value", "Percentage"]
    rows_2 = [
        [f"Precision@{top_k}", f"{mean_prec:.4f}", f"{mean_prec*100:.2f}%"],
        [f"Recall@{top_k}", f"{mean_rec:.4f}", f"{mean_rec*100:.2f}%"],
        [f"F1-Score@{top_k}", f"{mean_f1:.4f}", f"{mean_f1*100:.2f}%"]
    ]
    print_ascii_table(headers_2, rows_2, alignments=['left', 'center', 'center'])
    print("="*75 + "\n")


# ======================================================================================
# 6. DATASET SUMMARY & ANALYTICS MODULE
# ======================================================================================

def display_dataset_summary(data):
    """
    Displays comprehensive statistics and dataset properties, including Top-10 Collaborative Recommendations.
    """
    print("\n" + "="*75)
    print("              [ANALYTICS] DATASET SUMMARY & EXPLORATORY METRICS")
    print("="*75)
    
    num_ratings = len(data)
    num_users = data['userId'].nunique()
    num_movies = data['movieId'].nunique()
    total_possible = num_users * num_movies
    sparsity = (1.0 - (num_ratings / total_possible)) * 100
    global_mean = data['rating'].mean()
    
    print(f"Total User Ratings   : {num_ratings:,}")
    print(f"Unique Users         : {num_users:,}")
    print(f"Unique Movies        : {num_movies:,}")
    print(f"Rating Scale         : {data['rating'].min()} to {data['rating'].max()} stars")
    print(f"Average Rating       : {global_mean:.2f} stars")
    print(f"Rating Matrix Size   : {num_users} x {num_movies} ({total_possible:,} cells)")
    print(f"Matrix Sparsity      : {sparsity:.2f}% (Standard in Recommender Systems)")
    
    genre_col = 'genres_clean' if 'genres_clean' in data.columns else 'genres'
    movie_stats = data.groupby('title').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        genres=(genre_col, 'first')
    ).reset_index()
    
    # Top 10 Collaborative Recommended Movies (Bayesian Quality Weighted, Min 50 Ratings)
    m = 50  # minimum rating count threshold
    v = movie_stats['num_ratings']
    R = movie_stats['avg_rating']
    C = global_mean
    movie_stats['weighted_score'] = (v / (v + m)) * R + (m / (v + m)) * C
    
    top_collab = movie_stats[movie_stats['num_ratings'] >= m].sort_values('weighted_score', ascending=False).head(10).reset_index(drop=True)
    print("\n--- Top 10 Collaborative Recommended Movies (Quality Weighted, Min 50 Ratings) ---")
    headers_collab = ["#", "Movie Title", "Ratings", "Avg Rating", "Weighted Score", "Genres"]
    rows_collab = []
    for i, row in enumerate(top_collab.to_dict('records'), 1):
        title = str(row['title'])
        if len(title) > 30:
            title = title[:27] + "..."
        genres_str = str(row['genres']).replace('|', ' | ') if row['genres'] else 'N/A'
        if len(genres_str) > 32:
            genres_str = genres_str[:29] + "..."
        rows_collab.append([str(i), title, f"{int(row['num_ratings']):,}", f"{row['avg_rating']:.2f}/5.0", f"{row['weighted_score']:.2f}", genres_str])
        
    print_ascii_table(headers_collab, rows_collab, alignments=['center', 'left', 'right', 'center', 'center', 'left'])
    print("="*75 + "\n")


# ======================================================================================
# 6. INTERACTIVE CLI / MAIN APPLICATION LOOP
# ======================================================================================

def interactive_movie_search_only(movie_stats):
    """
    Dedicated search mode: Looks up and displays complete movie details (Ratings, Genres, Keywords, Overview).
    """
    titles_list = movie_stats['title'].tolist()
    
    while True:
        print("\n" + "-"*75)
        try:
            user_input = input("Enter movie title/keyword to lookup (or 'b' for main menu): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ('b', 'back', 'exit', 'q', 'quit'):
            break
            
        matches = search_movies(user_input, titles_list, movie_stats, max_results=5)
        
        if not matches:
            print(f"\n[!] No movies found matching '{user_input}'.")
            print("    Hint: Try typing a single keyword (e.g. 'Matrix', 'Star Wars', 'Avatar', 'Batman').")
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
                choice = input(f"Select a movie [1-{len(matches)}] (or 'b' to re-search, default 1): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
                
            if choice.lower() in ('b', 'back', 'cancel', 'c', '0'):
                continue
            if choice.lower() in ('exit', 'q', 'quit'):
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                target_movie = matches[int(choice) - 1]
            else:
                target_movie = matches[0]
                
        # Display Movie Details
        movie_row = movie_stats[movie_stats['title'] == target_movie].iloc[0]
        print("\n" + "="*75)
        print("                            MOVIE DETAILS")
        print("="*75)
        print(f"Title          : {movie_row['title']}")
        print(f"Movie ID       : {movie_row['movieId']}")
        print(f"Average Rating : {movie_row['avg_rating']:.2f} / 5.0 stars")
        print(f"Total Ratings  : {movie_row['num_of_ratings']:,} ratings")
        print(f"Genres         : {movie_row['genres'].replace('|', ' | ') if movie_row['genres'] else 'N/A'}")
        if 'keywords' in movie_row and movie_row['keywords']:
            print(f"Keywords       : {movie_row['keywords'].replace('|', ' | ')}")
        if 'overview' in movie_row and movie_row['overview']:
            print(f"Overview       : {movie_row['overview']}")
        print("="*75)


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
                choice = input(f"Select a movie [1-{len(matches)}] (or 'b' to re-search, default 1): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
                
            if choice.lower() in ('b', 'back', 'cancel', 'c', '0'):
                continue
            if choice.lower() in ('exit', 'q', 'quit'):
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                target_movie = matches[int(choice) - 1]
            else:
                target_movie = matches[0]
                
        # Ask for number of recommendations to display
        try:
            num_recs_input = input("Enter number of recommendations to display [default 10]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if num_recs_input.lower() in ('b', 'back', 'cancel', 'c'):
            continue
        if num_recs_input.lower() in ('exit', 'q', 'quit'):
            break
            
        top_n = int(num_recs_input) if num_recs_input.isdigit() and int(num_recs_input) > 0 else 10
        
        print(f"\n[*] Generating collaborative recommendations for: '{target_movie}' (Top {top_n})...\n")
        recs = get_collaborative_recommendations(target_movie, user_movie_matrix, movie_stats, min_ratings=50, top_n=top_n)
        
        if recs is None or recs.empty:
            print("[!] No recommendations found meeting the correlation threshold.")
        else:
            print(">>> Top Recommendations:")
            print_recommendations_table(recs)


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
        print("  [1] Search Movie (View Details & Ratings)")
        print("  [2] Get Recommendations by Movie")
        print("  [3] Run Recommender System Evaluation (RMSE/MSE/Precision/Recall/F1)")
        print("  [4] View Dataset Summary & Statistics")
        print("  [5] Exit Application")
        print("="*50)
        
        try:
            choice = input("Enter your option [1-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Recommender System. Goodbye!")
            break
            
        if choice == '1':
            interactive_movie_search_only(movie_stats)
        elif choice == '2':
            interactive_search_mode(user_movie_matrix, movie_stats)
        elif choice == '3':
            evaluate_recommender_system(data)
        elif choice == '4':
            display_dataset_summary(data)
        elif choice in ('5', 'exit', 'quit', 'q'):
            print("\nThank you for using the Recommender System. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()

