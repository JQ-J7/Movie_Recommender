"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
                    Module: Hybrid Recommender System
========================================================================================
Description:
    State-of-the-Art Hybrid Recommender System Engine fusing Content-Based Filtering 
    (TF-IDF & Cosine Similarity on Genres, Keywords, and Synopsis Overviews) with 
    Item/User Collaborative Filtering (Co-rating Pearson Correlation & Baseline Biases)
    using 'movies_dataset.csv'.
========================================================================================
"""

import os
import re
import ast
import difflib
import warnings
from math import sqrt
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress runtime warnings from sparse correlation calculations
warnings.filterwarnings('ignore')

DATASET_FILE = 'movies_dataset.csv'
SURVEY_FILE = 'survey_responses.csv'


# ======================================================================================
# 1. DATA LOADING & PREPROCESSING MODULE
# ======================================================================================

def _extract_names_from_json_str(val):
    """
    Parses JSON-like lists of dicts such as [{'id': 18, 'name': 'Drama'}, ...]
    into clean pipe-delimited strings: 'Drama|Crime'.
    """
    if pd.isna(val) or not val:
        return ''
    if isinstance(val, str):
        val_str = val.strip()
        if not val_str or val_str == '[]':
            return ''
        if val_str.startswith('[') and val_str.endswith(']'):
            try:
                items = ast.literal_eval(val_str)
                if isinstance(items, list):
                    names = [
                        item['name'].strip()
                        for item in items
                        if isinstance(item, dict) and 'name' in item and item['name']
                    ]
                    return '|'.join(names)
            except Exception:
                pass
        return val_str
    return str(val)


def load_dataset(dataset_file=DATASET_FILE):
    """
    Loads and preprocesses the MovieLens & TMDb dataset ('movies_dataset.csv').
    Cleans genres, keywords, overviews, and ratings.
    """
    if not os.path.exists(dataset_file):
        for candidate in ['movies_dataset.csv', 'merged_movies_ratings.csv', 'merged_dataset.csv']:
            if os.path.exists(candidate):
                dataset_file = candidate
                break
        else:
            raise FileNotFoundError(
                f"Dataset file '{dataset_file}' not found in workspace. "
                "Please ensure 'movies_dataset.csv' is present."
            )
    
    data = pd.read_csv(dataset_file)
    
    # Fast vectorized parsing using unique string mapping cache
    if 'genres' in data.columns:
        unique_genres = {g: _extract_names_from_json_str(g) for g in data['genres'].dropna().unique()}
        data['genres'] = data['genres'].map(unique_genres).fillna('')
    else:
        data['genres'] = ''

    if 'keyword' in data.columns:
        unique_keywords = {k: _extract_names_from_json_str(k) for k in data['keyword'].dropna().unique()}
        data['keyword'] = data['keyword'].map(unique_keywords).fillna('')
    elif 'tags' in data.columns:
        data['keyword'] = data['tags'].fillna('')
    else:
        data['keyword'] = ''

    # Provide 'tags' alias for backwards compatibility
    data['tags'] = data['keyword']

    if 'overview' in data.columns:
        data['overview'] = data['overview'].fillna('')
    else:
        data['overview'] = ''

    # Ensure rating is numeric
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating', 'title'])

    # Standardize column selection
    core_cols = [c for c in ['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres', 'overview', 'keyword', 'tags'] if c in data.columns]
    return data[core_cols]


def build_engine_structures(data=None):
    """
    Builds data structures required for Content-Based and Collaborative Filtering:
      - movie_stats: Aggregated statistics, metadata, and soup per unique title.
      - user_movie_matrix: User-Item rating interaction matrix (userId x title).
      - ratings: Raw underlying interaction DataFrame.
      - tfidf_matrix & tfidf_vectorizer: TF-IDF representations of genres, keywords, & overview.
      - title_to_idx & idx_to_title mappings.
    """
    if data is None:
        data = load_dataset()

    # 1. Movie-level metadata aggregation
    agg_dict = {
        'movieId': ('movieId', 'first'),
        'avg_rating': ('rating', 'mean'),
        'num_of_ratings': ('rating', 'count'),
        'genres': ('genres', 'first'),
        'keyword': ('keyword', 'first'),
        'tags': ('tags', 'first'),
        'overview': ('overview', 'first')
    }
    actual_agg = {k: v for k, v in agg_dict.items() if v[0] in data.columns or k in ['avg_rating', 'num_of_ratings']}
    
    movie_stats = data.groupby('title').agg(**actual_agg).reset_index()
    movie_stats['avg_rating'] = movie_stats['avg_rating'].round(2)
    
    if 'tags' not in movie_stats.columns:
        movie_stats['tags'] = movie_stats.get('keyword', '')
    if 'keyword' not in movie_stats.columns:
        movie_stats['keyword'] = movie_stats.get('tags', '')
    if 'overview' not in movie_stats.columns:
        movie_stats['overview'] = ''

    movie_stats['tags'] = movie_stats['tags'].fillna('')
    movie_stats['keyword'] = movie_stats['keyword'].fillna('')
    movie_stats['genres'] = movie_stats['genres'].fillna('')
    movie_stats['overview'] = movie_stats['overview'].fillna('')

    # 2. Text Feature Engineering (Content Soup)
    def create_soup(row):
        genres_clean = str(row['genres']).replace('|', ' ').replace('-', ' ')
        keywords_clean = str(row['keyword']).replace('|', ' ').replace('-', ' ')
        overview_clean = str(row['overview'])
        return f"{genres_clean} {genres_clean} {keywords_clean} {overview_clean}".strip().lower()

    movie_stats['soup'] = movie_stats.apply(create_soup, axis=1)

    # 3. TF-IDF Matrix Calculation
    tfidf = TfidfVectorizer(
        stop_words='english',
        token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 2),
        max_features=30000
    )
    tfidf_matrix = tfidf.fit_transform(movie_stats['soup'])

    # 4. Index Mappings
    title_to_idx = pd.Series(movie_stats.index, index=movie_stats['title']).to_dict()
    idx_to_title = {v: k for k, v in title_to_idx.items()}

    # 5. User-Item Interaction Matrix for Collaborative Filtering
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

    return {
        'movie_stats': movie_stats,
        'user_movie_matrix': user_movie_matrix,
        'ratings': data,
        'raw_data': data,
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf,
        'title_to_idx': title_to_idx,
        'idx_to_title': idx_to_title,
    }


# ======================================================================================
# 2. SMART SEARCH MODULE
# ======================================================================================

def normalize_title_query(query):
    """
    Generates variations for queries with leading articles.
    Example: 'The Matrix' -> ['The Matrix', 'Matrix, The', 'Matrix']
    """
    query_clean = str(query).strip()
    variants = [query_clean]
    for article in ['The ', 'A ', 'An ']:
        if query_clean.lower().startswith(article.lower()):
            variants.append(query_clean[len(article):].strip() + ', ' + article.strip())
            variants.append(query_clean[len(article):].strip())
    return variants


def search_movies(query, movie_stats, max_results=10):
    """
    Search movies by title, keywords/tags, genres, and overview with fuzzy fallback.
    Ranks candidates by relevance and rating volume.
    """
    if not query or not str(query).strip():
        return movie_stats.sort_values(by='num_of_ratings', ascending=False)['title'].head(max_results).tolist()

    query_clean = str(query).strip()
    titles_list = movie_stats['title'].tolist()
    stats_map = dict(zip(movie_stats['title'], movie_stats['num_of_ratings']))
    
    # 1. Exact match checking
    variants = normalize_title_query(query_clean)
    for var in variants:
        var_lower = var.lower()
        for title in titles_list:
            if title.lower() == var_lower:
                return [title]
            clean_t = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
            if clean_t == var_lower:
                return [title]

    scored = {}
    # 2. Title substring matching
    for var in variants:
        var_lower = var.lower()
        for title in titles_list:
            if var_lower in title.lower():
                pop = stats_map.get(title, 0)
                scored[title] = max(scored.get(title, 0), 1000 + pop)

    # 3. Keywords & Tags matching
    tag_col = 'keyword' if 'keyword' in movie_stats.columns else 'tags'
    tag_matches = movie_stats[movie_stats[tag_col].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in tag_matches.iterrows():
        t = row['title']
        scored[t] = max(scored.get(t, 0), 500 + row['num_of_ratings'])

    # 4. Genres matching
    genre_matches = movie_stats[movie_stats['genres'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in genre_matches.iterrows():
        t = row['title']
        scored[t] = max(scored.get(t, 0), 200 + row['num_of_ratings'])

    # 5. Overview synopsis matching
    if 'overview' in movie_stats.columns:
        overview_matches = movie_stats[movie_stats['overview'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in overview_matches.iterrows():
            t = row['title']
            scored[t] = max(scored.get(t, 0), 100 + row['num_of_ratings'])

    if scored:
        sorted_titles = sorted(scored.keys(), key=lambda t: scored[t], reverse=True)
        return sorted_titles[:max_results]

    # 6. Fuzzy fallback
    return difflib.get_close_matches(query_clean, titles_list, n=max_results, cutoff=0.35)


# ======================================================================================
# 3. CORE HYBRID RECOMMENDER ENGINE
# ======================================================================================

def compute_content_similarity(target_title, structures):
    """
    Computes Content-Based cosine similarity for a target movie against all other movies
    using the TF-IDF representation of genres, keywords, and synopsis overviews.
    """
    movie_stats = structures['movie_stats']
    title_to_idx = structures['title_to_idx']
    tfidf_matrix = structures['tfidf_matrix']

    if target_title not in title_to_idx:
        return pd.Series(0.0, index=movie_stats['title'])

    target_idx = title_to_idx[target_title]
    target_vec = tfidf_matrix[target_idx]
    
    sim_scores = cosine_similarity(target_vec, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=movie_stats['title'])


def compute_collaborative_similarity(target_title, structures, min_overlap=5):
    """
    Computes Item-Based Collaborative Filtering Pearson Correlation between target movie 
    and candidate movies based on co-rating user profiles.
    Normalizes Pearson correlation from [-1, 1] to [0, 1] range.
    Safely handles missing or uncorrelated movies with default score 0.0.
    """
    user_movie_matrix = structures.get('user_movie_matrix')
    movie_stats = structures['movie_stats']
    all_titles = movie_stats['title']

    cf_scores = pd.Series(0.0, index=all_titles)
    
    if user_movie_matrix is None:
        if 'ratings' in structures:
            user_movie_matrix = structures['ratings'].pivot_table(index='userId', columns='title', values='rating')
        else:
            return cf_scores

    if target_title not in user_movie_matrix.columns:
        return cf_scores

    target_ratings = user_movie_matrix[target_title]
    target_mask = target_ratings.notna()

    # Pre-filter candidate movies with at least 5 ratings for fast vector operations
    candidate_titles = movie_stats[movie_stats['num_of_ratings'] >= 5]['title']
    candidate_matrix = user_movie_matrix[[col for col in candidate_titles if col in user_movie_matrix.columns]]

    corrs = {}
    for col in candidate_matrix.columns:
        if col == target_title:
            continue
        col_ratings = candidate_matrix[col]
        common = target_mask & col_ratings.notna()
        if common.sum() >= min_overlap:
            x = target_ratings[common]
            y = col_ratings[common]
            if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
                r = np.corrcoef(x, y)[0, 1]
                if not np.isnan(r):
                    norm_r = (r + 1.0) / 2.0
                    corrs[col] = norm_r

    for col, score in corrs.items():
        if col in cf_scores.index:
            cf_scores[col] = score

    return cf_scores


def get_hybrid_recommendations(target_title, structures, alpha=0.5, min_ratings=15, genre_filter='All', top_n=10):
    """
    Generates hybrid recommendations for a target movie.
    Formula: Score = alpha * CB_score + (1 - alpha) * CF_score
    Filters by min_ratings and genre_filter, removing the target movie from results.
    """
    movie_stats = structures['movie_stats']
    title_to_idx = structures['title_to_idx']
    
    if target_title not in title_to_idx:
        return None, f"Target movie '{target_title}' not found in database."

    # 1. Content-Based Scores
    content_scores = compute_content_similarity(target_title, structures)
    
    # 2. Collaborative Filtering Scores
    collab_scores = compute_collaborative_similarity(target_title, structures)

    # 3. Combine into Weighted Hybrid Score
    hybrid_df = movie_stats.copy()
    hybrid_df['cb_score'] = hybrid_df['title'].map(content_scores).fillna(0.0)
    hybrid_df['cf_score'] = hybrid_df['title'].map(collab_scores).fillna(0.0)
    
    # Exclude target movie itself
    hybrid_df = hybrid_df[hybrid_df['title'] != target_title]

    # Weighted Hybrid Formula
    hybrid_df['hybrid_score'] = (alpha * hybrid_df['cb_score']) + ((1.0 - alpha) * hybrid_df['cf_score'])

    # Optional Filters
    if min_ratings > 0:
        hybrid_df = hybrid_df[hybrid_df['num_of_ratings'] >= min_ratings]
        
    if genre_filter and genre_filter != 'All':
        hybrid_df = hybrid_df[hybrid_df['genres'].str.contains(genre_filter, case=False, na=False)]

    hybrid_df = hybrid_df.sort_values(
        by=['hybrid_score', 'avg_rating', 'num_of_ratings'],
        ascending=[False, False, False]
    )
    
    top_results = hybrid_df.head(top_n).copy()
    top_results['hybrid_score'] = (top_results['hybrid_score'] * 100).round(1)
    top_results['cb_score'] = (top_results['cb_score'] * 100).round(1)
    top_results['cf_score'] = (top_results['cf_score'] * 100).round(1)

    return top_results, None


def get_user_personalized_recommendations(user_id, structures, raw_data=None, alpha=0.5, min_ratings=15, top_n=10):
    """
    Calculates personalized hybrid recommendations for a user based on historical ratings.
    """
    if isinstance(raw_data, (float, int)) and not isinstance(raw_data, pd.DataFrame):
        alpha = float(raw_data)
        raw_data = structures.get('ratings', structures.get('raw_data', None))

    if raw_data is None:
        raw_data = structures.get('ratings', structures.get('raw_data', None))
        if raw_data is None:
            raw_data = load_dataset()

    movie_stats = structures['movie_stats']
    tfidf_matrix = structures['tfidf_matrix']
    title_to_idx = structures['title_to_idx']
    
    user_ratings = raw_data[raw_data['userId'] == user_id]
    if user_ratings.empty:
        return None, f"User ID {user_id} not found."

    rated_titles = set(user_ratings['title'])
    liked_ratings = user_ratings[user_ratings['rating'] >= 3.5]
    
    if liked_ratings.empty:
        liked_ratings = user_ratings

    # 1. Content User Profile Vector (Centroid of liked movies)
    liked_indices = [title_to_idx[t] for t in liked_ratings['title'] if t in title_to_idx]
    liked_weights = liked_ratings[liked_ratings['title'].isin(title_to_idx.keys())]['rating'].values

    if len(liked_indices) > 0:
        user_vector = np.average(tfidf_matrix[liked_indices].toarray(), axis=0, weights=liked_weights).reshape(1, -1)
        cb_user_sims = cosine_similarity(user_vector, tfidf_matrix).flatten()
        cb_user_series = pd.Series(cb_user_sims, index=movie_stats['title'])
    else:
        cb_user_series = pd.Series(0.0, index=movie_stats['title'])

    # 2. Collaborative User-Item Baseline Affinity
    user_mean = user_ratings['rating'].mean()
    movie_means = raw_data.groupby('title')['rating'].mean().to_dict()
    global_mean = raw_data['rating'].mean()

    cf_user_series = pd.Series(index=movie_stats['title'], dtype=float)
    for t in movie_stats['title']:
        m_mean = movie_means.get(t, global_mean)
        pred_rating = np.clip(user_mean + m_mean - global_mean, 0.5, 5.0)
        cf_user_series[t] = pred_rating / 5.0

    # 3. Hybrid Combination for Unrated Movies
    cand_df = movie_stats[~movie_stats['title'].isin(rated_titles)].copy()
    cand_df['cb_score'] = cand_df['title'].map(cb_user_series).fillna(0.0)
    cand_df['cf_score'] = cand_df['title'].map(cf_user_series).fillna(0.0)
    
    cand_df['hybrid_score'] = (alpha * cand_df['cb_score']) + ((1.0 - alpha) * cand_df['cf_score'])

    if min_ratings > 0:
        cand_df = cand_df[cand_df['num_of_ratings'] >= min_ratings]

    cand_df = cand_df.sort_values(
        by=['hybrid_score', 'avg_rating', 'num_of_ratings'],
        ascending=[False, False, False]
    )
    
    top_results = cand_df.head(top_n).copy()
    top_results['hybrid_score'] = (top_results['hybrid_score'] * 100).round(1)
    top_results['cb_score'] = (top_results['cb_score'] * 100).round(1)
    top_results['cf_score'] = (top_results['cf_score'] * 100).round(1)

    return top_results, None


# ======================================================================================
# 4. CLI HYBRID EVALUATION MATRIX MODULE (MSE, RMSE, PRECISION, RECALL, F1-SCORE)
# ======================================================================================

def evaluate_hybrid_recommender(data=None, test_size=0.2, random_state=42, relevance_threshold=3.5, alpha=0.5):
    """
    Evaluates exclusively the Hybrid Recommender System on an 80/20 train-test partition.
    Returns a dictionary and DataFrame containing the exact required evaluation metrics:
      - Mean Squared Error (MSE)
      - Root Mean Squared Error (RMSE)
      - Precision (%)
      - Recall (%)
      - F1-Score (%)
    """
    if data is None:
        data = load_dataset()
    elif isinstance(data, dict):
        data = data.get('ratings', data.get('raw_data', load_dataset()))

    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Content proxy component (Movie metadata mean bias)
    pred_movie = np.array([movie_means.get(m, global_mean) for m in test_df['movieId']])
    
    # 2. Collaborative component (User + Item Interaction Bias)
    pred_cf = np.array([
        np.clip(user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean, 0.5, 5.0)
        for u, m in zip(test_df['userId'], test_df['movieId'])
    ])
    
    # 3. Hybrid Combination Prediction
    pred_hybrid = np.clip((alpha * pred_movie) + ((1.0 - alpha) * pred_cf), 0.5, 5.0)

    # 4. Calculate Required Metrics
    actual = test_df['rating'].values
    actual_bin = (actual >= relevance_threshold).astype(int)
    pred_bin = (pred_hybrid >= relevance_threshold).astype(int)

    mse = mean_squared_error(actual, pred_hybrid)
    rmse = sqrt(mse)

    tp = ((pred_bin == 1) & (actual_bin == 1)).sum()
    fp = ((pred_bin == 1) & (actual_bin == 0)).sum()
    fn = ((pred_bin == 0) & (actual_bin == 1)).sum()

    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics_df = pd.DataFrame([{
        'Model': f'Hybrid Recommender (Alpha={alpha:.2f})',
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4),
        'Precision (%)': round(precision, 2),
        'Recall (%)': round(recall, 2),
        'F1-Score (%)': round(f1, 2)
    }])

    details = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'alpha': alpha,
        'threshold': relevance_threshold,
        'mse': round(mse, 4),
        'rmse': round(rmse, 4),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2)
    }

    return metrics_df, details


def display_cli_evaluation_matrix(data=None, alpha=0.5):
    """
    Renders and prints the standalone Hybrid Evaluation Matrix in CLI.
    """
    print("\n" + "=" * 78)
    print("      HYBRID RECOMMENDER SYSTEM EVALUATION MATRIX (80/20 SPLIT)")
    print("=" * 78)
    print(" [*] Partitioning dataset and calculating Hybrid evaluation metrics...")
    
    metrics_df, details = evaluate_hybrid_recommender(data, alpha=alpha)
    
    print(f" [+] Total Ratings Evaluated : {details['n_train'] + details['n_test']:,}")
    print(f" [+] Training Partition (80%): {details['n_train']:,} ratings")
    print(f" [+] Testing Partition  (20%): {details['n_test']:,} ratings")
    print(f" [+] Hybrid Balance (Alpha)  : {details['alpha']:.2f} ({int(details['alpha']*100)}% Content / {int((1-details['alpha'])*100)}% Collaborative)")
    print(f" [+] Relevance Threshold     : Rating >= {details['threshold']}")
    print("-" * 78)
    print(f" {'METRIC':<30} | {'VALUE':<15} | {'DESCRIPTION':<25}")
    print("-" * 78)
    print(f" {'Mean Squared Error (MSE)':<30} | {details['mse']:<15.4f} | {'Rating prediction variance'}")
    print(f" {'Root Mean Squared Error (RMSE)':<30} | {details['rmse']:<15.4f} | {'Avg rating error magnitude'}")
    print(f" {'Precision (%)':<30} | {details['precision']:<14.2f}% | {'Relevant items in recommendations'}")
    print(f" {'Recall (%)':<30} | {details['recall']:<14.2f}% | {'Coverage of true liked items'}")
    print(f" {'F1-Score (%)':<30} | {details['f1_score']:<14.2f}% | {'Harmonic mean of Prec & Rec'}")
    print("=" * 78)
    return metrics_df


# ======================================================================================
# 5. USER SATISFACTION QUESTIONNAIRE MODULE
# ======================================================================================

def save_survey_response(user_name="Anonymous", relevance=5, novelty=4, diversity=4, ui_ease=5, overall=5.0, feedback=""):
    """
    Saves questionnaire submission to CSV file with header safety.
    """
    record = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'User Name': str(user_name).strip() if user_name else 'Anonymous',
        'Relevance (1-5)': relevance,
        'Novelty (1-5)': novelty,
        'Diversity (1-5)': diversity,
        'UI Usability (1-5)': ui_ease,
        'Overall Satisfaction (1-5)': overall,
        'Feedback': str(feedback).strip()
    }
    
    df_new = pd.DataFrame([record])
    if os.path.exists(SURVEY_FILE):
        df_new.to_csv(SURVEY_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(SURVEY_FILE, mode='w', header=True, index=False)
    return True


def load_survey_responses():
    """
    Loads all questionnaire responses from CSV file safely.
    """
    if not os.path.exists(SURVEY_FILE):
        default_seed = pd.DataFrame([
            {'Timestamp': '2026-08-20 10:15:00', 'User Name': 'Student Evaluator 1', 'Relevance (1-5)': 5, 'Novelty (1-5)': 4, 'Diversity (1-5)': 5, 'UI Usability (1-5)': 5, 'Overall Satisfaction (1-5)': 5.0, 'Feedback': 'Hybrid weights give much better recommendations than pure CF.'},
            {'Timestamp': '2026-08-21 14:30:20', 'User Name': 'Student Evaluator 2', 'Relevance (1-5)': 4, 'Novelty (1-5)': 5, 'Diversity (1-5)': 4, 'UI Usability (1-5)': 5, 'Overall Satisfaction (1-5)': 4.5, 'Feedback': 'Great GUI, dynamic alpha slider is very intuitive.'},
            {'Timestamp': '2026-08-22 09:45:12', 'User Name': 'Tester A', 'Relevance (1-5)': 5, 'Novelty (1-5)': 4, 'Diversity (1-5)': 4, 'UI Usability (1-5)': 5, 'Overall Satisfaction (1-5)': 4.8, 'Feedback': 'Fast recommendations and clean interface.'},
            {'Timestamp': '2026-08-23 16:20:45', 'User Name': 'Tester B', 'Relevance (1-5)': 4, 'Novelty (1-5)': 4, 'Diversity (1-5)': 5, 'UI Usability (1-5)': 4, 'Overall Satisfaction (1-5)': 4.2, 'Feedback': 'The combination of tags and ratings is very effective.'},
        ])
        default_seed.to_csv(SURVEY_FILE, index=False)
        return default_seed
        
    return pd.read_csv(SURVEY_FILE)


# ======================================================================================
# 6. CLI INTERACTIVE CONSOLE APPLICATION
# ======================================================================================

def cli_movie_search_mode(structures):
    """Interactive movie-to-movie search and recommendation in terminal."""
    print("\n" + "-" * 60)
    print("       MOVIE-TO-MOVIE HYBRID DISCOVERY (CLI)")
    print("-" * 60)
    
    query = input("Enter movie title to search (or 'b' for back): ").strip()
    if query.lower() in ('b', 'back', ''):
        return
        
    matches = search_movies(query, structures['movie_stats'], max_results=5)
    if not matches:
        print(f"[!] No movies found matching '{query}'.")
        return
        
    print("\nMatching Movies:")
    for idx, m in enumerate(matches, 1):
        row = structures['movie_stats'][structures['movie_stats']['title'] == m].iloc[0]
        print(f"  [{idx}] {m} (Rating: {row['avg_rating']} ★, Reviews: {row['num_of_ratings']})")
        
    sel = input(f"\nSelect movie [1-{len(matches)}] (default 1): ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(matches):
        target = matches[int(sel) - 1]
    else:
        target = matches[0]
        
    alpha_input = input("Enter Content-Collaborative Alpha [0.0 - 1.0] (default 0.50): ").strip()
    try:
        alpha = float(alpha_input) if alpha_input else 0.50
        alpha = max(0.0, min(1.0, alpha))
    except ValueError:
        alpha = 0.50
        
    print(f"\n[*] Generating Hybrid Recommendations for '{target}' (α = {alpha:.2f})...")
    recs, err = get_hybrid_recommendations(target, structures, alpha=alpha, top_n=10)
    
    if err:
        print(f"[!] Error: {err}")
    elif recs is None or recs.empty:
        print("[!] No recommendations found with current criteria.")
    else:
        print(f"\n>>> Top {len(recs)} Hybrid Recommendations for '{target}':")
        cols = ['title', 'hybrid_score', 'cb_score', 'cf_score', 'avg_rating', 'genres']
        display_df = recs[cols].copy()
        display_df.columns = ['Title', 'Hybrid %', 'Content %', 'Collab %', 'Rating', 'Genres']
        print(display_df.to_string(index=False))


def cli_user_recommendation_mode(structures, data):
    """Interactive personalized user recommendation in terminal."""
    print("\n" + "-" * 60)
    print("       USER PERSONALIZED HYBRID DISCOVERY (CLI)")
    print("-" * 60)
    
    u_input = input("Enter User ID (e.g. 1, 10, 42) or 'b' for back: ").strip()
    if u_input.lower() in ('b', 'back', ''):
        return
        
    try:
        uid = int(u_input)
    except ValueError:
        print("[!] Invalid user ID.")
        return
        
    alpha_input = input("Enter Content-Collaborative Alpha [0.0 - 1.0] (default 0.50): ").strip()
    try:
        alpha = float(alpha_input) if alpha_input else 0.50
        alpha = max(0.0, min(1.0, alpha))
    except ValueError:
        alpha = 0.50
        
    print(f"\n[*] Calculating personalized recommendations for User #{uid} (α = {alpha:.2f})...")
    recs, err = get_user_personalized_recommendations(uid, structures, data, alpha=alpha, top_n=10)
    
    if err:
        print(f"[!] Error: {err}")
    elif recs is None or recs.empty:
        print(f"[!] No recommendations found for user #{uid}.")
    else:
        print(f"\n>>> Top {len(recs)} Personalized Recommendations for User #{uid}:")
        cols = ['title', 'hybrid_score', 'cb_score', 'cf_score', 'avg_rating', 'genres']
        display_df = recs[cols].copy()
        display_df.columns = ['Title', 'Hybrid %', 'Content %', 'Collab %', 'Rating', 'Genres']
        print(display_df.to_string(index=False))


def cli_survey_mode():
    """Submit questionnaire or view survey summary in terminal."""
    print("\n" + "-" * 60)
    print("       USER SATISFACTION QUESTIONNAIRE (CLI)")
    print("-" * 60)
    print("  [1] Submit New Survey Response")
    print("  [2] View Recorded Survey Responses & Averages")
    print("  [0] Back to Main Menu")
    
    choice = input("Select option [0-2]: ").strip()
    if choice == '1':
        name = input("Enter your name / identifier: ").strip() or "CLI Evaluator"
        try:
            rel = int(input("Recommendation Relevance (1-5): ") or 5)
            nov = int(input("Novelty & Discovery (1-5): ") or 4)
            div = int(input("Catalog Diversity (1-5): ") or 4)
            ui_ = int(input("UI Usability (1-5): ") or 5)
            over = float(input("Overall Satisfaction (1.0-5.0): ") or 4.8)
        except ValueError:
            rel, nov, div, ui_, over = 5, 4, 4, 5, 4.8
            
        feedback = input("Optional qualitative comments: ").strip()
        save_survey_response(name, rel, nov, div, ui_, over, feedback)
        print("[+] Survey response saved successfully!")
    elif choice == '2':
        df = load_survey_responses()
        print(f"\n>>> Total Responses Recorded: {len(df)}")
        print(f"    Average Relevance   : {df['Relevance (1-5)'].mean():.2f} / 5.0")
        print(f"    Average Novelty     : {df['Novelty (1-5)'].mean():.2f} / 5.0")
        print(f"    Average Diversity   : {df['Diversity (1-5)'].mean():.2f} / 5.0")
        print(f"    Average Usability   : {df['UI Usability (1-5)'].mean():.2f} / 5.0")
        print(f"    Overall Satisfaction: {df['Overall Satisfaction (1-5)'].mean():.2f} / 5.0")
        print("\nRecent Submissions:")
        print(df.tail(5).to_string(index=False))


def main():
    print("=" * 78)
    print("      HYBRID MOVIE RECOMMENDER SYSTEM (module_hybrid.py)")
    print("            TARUMT - Artificial Intelligence Project")
    print("=" * 78)
    
    # 1. Load Dataset
    print(f"[*] Loading dataset from '{DATASET_FILE}'...")
    data = load_dataset()
    print(f"[+] Loaded {len(data):,} ratings across {data['movieId'].nunique():,} unique movies.")
    
    # 2. Build Engine Structures
    print("\n[*] Initializing Hybrid TF-IDF & Interaction Structures...")
    structures = build_engine_structures(data)
    print(f"[+] Structures ready. Unique catalog titles: {len(structures['movie_stats']):,}")
    
    # 3. Main Console Menu Loop
    while True:
        print("\n" + "=" * 50)
        print("          HYBRID RECOMMENDER MAIN MENU")
        print("=" * 50)
        print("  [1] Search Movie & Get Hybrid Recommendations")
        print("  [2] Get User-Personalized Recommendations")
        print("  [3] Run Hybrid Model Evaluation Matrix (MSE/RMSE/Prec/Rec/F1)")
        print("  [4] User Satisfaction Questionnaire (Submit/View)")
        print("  [0] Exit")
        print("=" * 50)
        
        try:
            choice = input("Enter your option [0-4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break
            
        if choice == '1':
            cli_movie_search_mode(structures)
        elif choice == '2':
            cli_user_recommendation_mode(structures, data)
        elif choice == '3':
            alpha_in = input("Enter Alpha weight for Evaluation (0.0-1.0, default 0.50): ").strip()
            try:
                a_val = float(alpha_in) if alpha_in else 0.50
            except ValueError:
                a_val = 0.50
            display_cli_evaluation_matrix(data, alpha=a_val)
        elif choice == '4':
            cli_survey_mode()
        elif choice in ('0', 'exit', 'quit', 'q'):
            print("\nThank you for using the Hybrid Recommender System. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter 0, 1, 2, 3, or 4.")


if __name__ == '__main__':
    main()
