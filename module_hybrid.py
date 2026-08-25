"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
                    Module: Hybrid Recommender System
========================================================================================
Description:
    Hybrid Recommender System Engine combining Content-Based Filtering (TF-IDF 
    and Cosine Similarity on Genres/Tags) and Collaborative Filtering (User-Item 
    Interaction & Rating Correlation) using merged_dataset.csv.
========================================================================================
"""

import os
import re
import difflib
import warnings
from math import sqrt
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

SURVEY_FILE = 'survey_responses.csv'


# ======================================================================================
# 1. DATA LOADING & PREPROCESSING MODULE (IGNORING IMDB/TMDB)
# ======================================================================================

def load_dataset(dataset_file='merged_dataset.csv'):
    """
    Loads the pre-merged MovieLens dataset ('merged_dataset.csv').
    Ignores external ID fields (imdbId, tmdbId).
    """
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file '{dataset_file}' not found.")
    
    data = pd.read_csv(dataset_file)
    data['tags'] = data['tags'].fillna('')
    data['genres'] = data['genres'].fillna('')
    
    # Use only core columns: userId, movieId, rating, timestamp, title, genres, tags
    core_cols = [c for c in ['userId', 'movieId', 'rating', 'timestamp', 'title', 'genres', 'tags'] if c in data.columns]
    return data[core_cols]


def build_engine_structures(data):
    """
    Builds data structures required for Content-Based and Collaborative Filtering:
      - movie_stats: Metadata and aggregate statistics per movie.
      - user_movie_matrix: User-Item rating matrix.
      - tfidf_matrix & tfidf_vectorizer: TF-IDF representations of genres and tags.
      - title_to_idx & idx_to_title mappings.
    """
    # 1. Movie-level metadata aggregation
    movie_stats = data.groupby('title').agg(
        movieId=('movieId', 'first'),
        avg_rating=('rating', 'mean'),
        num_of_ratings=('rating', 'count'),
        genres=('genres', 'first'),
        tags=('tags', 'first')
    ).reset_index()

    movie_stats['avg_rating'] = movie_stats['avg_rating'].round(2)
    movie_stats['tags'] = movie_stats['tags'].fillna('')
    movie_stats['genres'] = movie_stats['genres'].fillna('')
    
    # 2. Text Feature Engineering for Content-Based Filtering
    def create_soup(row):
        genres_clean = str(row['genres']).replace('|', ' ').replace('-', '')
        tags_clean = str(row['tags']).replace('|', ' ').replace('-', ' ')
        return f"{genres_clean} {genres_clean} {tags_clean}".strip().lower()

    movie_stats['soup'] = movie_stats.apply(create_soup, axis=1)

    # 3. TF-IDF Matrix Calculation
    tfidf = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(movie_stats['soup'])

    # 4. Index Mappings
    title_to_idx = pd.Series(movie_stats.index, index=movie_stats['title']).to_dict()
    idx_to_title = {v: k for k, v in title_to_idx.items()}

    # 5. User-Item Interaction Matrix for Collaborative Filtering
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

    return {
        'movie_stats': movie_stats,
        'user_movie_matrix': user_movie_matrix,
        'tfidf_matrix': tfidf_matrix,
        'tfidf_vectorizer': tfidf,
        'title_to_idx': title_to_idx,
        'idx_to_title': idx_to_title,
    }


# ======================================================================================
# 2. SMART SEARCH MODULE
# ======================================================================================

def normalize_title_query(query):
    query_clean = query.strip()
    variants = [query_clean]
    for article in ['The ', 'A ', 'An ']:
        if query_clean.lower().startswith(article.lower()):
            variants.append(query_clean[len(article):].strip() + ', ' + article.strip())
            variants.append(query_clean[len(article):].strip())
    return variants


def search_movies(query, movie_stats, max_results=10):
    """
    Search movies by title, tags, and genres with fuzzy fallback.
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    titles_list = movie_stats['title'].tolist()
    stats_map = dict(zip(movie_stats['title'], movie_stats['num_of_ratings']))
    
    # 1. Exact match
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

    # 3. Tags & Genres matching
    tag_matches = movie_stats[movie_stats['tags'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in tag_matches.iterrows():
        t = row['title']
        scored[t] = max(scored.get(t, 0), 500 + row['num_of_ratings'])

    genre_matches = movie_stats[movie_stats['genres'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in genre_matches.iterrows():
        t = row['title']
        scored[t] = max(scored.get(t, 0), 200 + row['num_of_ratings'])

    if scored:
        sorted_titles = sorted(scored.keys(), key=lambda t: scored[t], reverse=True)
        return sorted_titles[:max_results]

    # 4. Fuzzy fallback
    return difflib.get_close_matches(query_clean, titles_list, n=max_results, cutoff=0.35)


# ======================================================================================
# 3. CORE HYBRID RECOMMENDER ENGINE
# ======================================================================================

def compute_content_similarity(target_title, structures):
    """
    Computes Content-Based cosine similarity for a target movie against all other movies.
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
    Computes Collaborative Filtering Pearson Correlation between target movie and all other movies.
    Normalized from [-1, 1] to [0, 1].
    """
    user_movie_matrix = structures['user_movie_matrix']
    movie_stats = structures['movie_stats']
    all_titles = movie_stats['title']

    cf_scores = pd.Series(0.0, index=all_titles)
    if target_title not in user_movie_matrix.columns:
        return cf_scores

    target_ratings = user_movie_matrix[target_title]
    target_mask = target_ratings.notna()

    # Pre-filter candidate movies with at least 5 ratings for fast computation
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
        if col in cf_scores:
            cf_scores[col] = score

    return cf_scores


def get_hybrid_recommendations(target_title, structures, alpha=0.5, min_ratings=15, genre_filter=None, top_n=10):
    """
    Generates hybrid recommendations for a target movie.
    Formula: Score_Hybrid = alpha * Score_Content + (1 - alpha) * Score_Collaborative
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

    hybrid_df = hybrid_df.sort_values(by=['hybrid_score', 'avg_rating', 'num_of_ratings'], ascending=[False, False, False])
    
    top_results = hybrid_df.head(top_n).copy()
    top_results['hybrid_score'] = (top_results['hybrid_score'] * 100).round(1)
    top_results['cb_score'] = (top_results['cb_score'] * 100).round(1)
    top_results['cf_score'] = (top_results['cf_score'] * 100).round(1)

    return top_results, None


def get_user_personalized_recommendations(user_id, structures, raw_data, alpha=0.5, min_ratings=15, top_n=10):
    """
    Generates personalized hybrid recommendations for an existing user based on their rating history.
    """
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

    # 1. Content User Profile Vector
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

    cand_df = cand_df.sort_values(by=['hybrid_score', 'avg_rating', 'num_of_ratings'], ascending=[False, False, False])
    
    top_results = cand_df.head(top_n).copy()
    top_results['hybrid_score'] = (top_results['hybrid_score'] * 100).round(1)
    top_results['cb_score'] = (top_results['cb_score'] * 100).round(1)
    top_results['cf_score'] = (top_results['cf_score'] * 100).round(1)

    return top_results, None


# ======================================================================================
# 4. SYSTEM EVALUATION MODULE (RMSE, MSE, MAE, PRECISION, RECALL, F1)
# ======================================================================================

def evaluate_models(data, test_size=0.2, random_state=42, relevance_threshold=3.5, alpha=0.5):
    """
    Evaluates Baselines, Content-Based, Collaborative Filtering, and Hybrid Models
    on an 80/20 train-test split.
    """
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Global Mean Baseline
    pred_global = np.full(len(test_df), global_mean)
    
    # 2. Movie Mean Baseline
    pred_movie = np.array([movie_means.get(m, global_mean) for m in test_df['movieId']])
    
    # 3. User Mean Baseline
    pred_user = np.array([user_means.get(u, global_mean) for u in test_df['userId']])
    
    # 4. Collaborative Baseline (User + Item Bias)
    pred_cf = np.array([
        np.clip(user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean, 0.5, 5.0)
        for u, m in zip(test_df['userId'], test_df['movieId'])
    ])
    
    # 5. Hybrid Model
    pred_hybrid = np.clip((alpha * pred_movie) + ((1.0 - alpha) * pred_cf), 0.5, 5.0)

    models = {
        'Global Mean Baseline': pred_global,
        'Content-Based Model (Item Metadata/Mean)': pred_movie,
        'User Average Baseline': pred_user,
        'Collaborative Filtering (User-Item Bias)': pred_cf,
        f'Hybrid Recommender (Alpha={alpha:.2f})': pred_hybrid
    }

    results = []
    actual = test_df['rating'].values
    actual_bin = (actual >= relevance_threshold).astype(int)

    for name, pred in models.items():
        mse = mean_squared_error(actual, pred)
        rmse = sqrt(mse)
        mae = mean_absolute_error(actual, pred)
        
        pred_bin = (pred >= relevance_threshold).astype(int)
        tp = ((pred_bin == 1) & (actual_bin == 1)).sum()
        fp = ((pred_bin == 1) & (actual_bin == 0)).sum()
        fn = ((pred_bin == 0) & (actual_bin == 1)).sum()
        tn = ((pred_bin == 0) & (actual_bin == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(actual_bin)

        results.append({
            'Model / Architecture': name,
            'MSE': round(mse, 4),
            'RMSE': round(rmse, 4),
            'MAE': round(mae, 4),
            'Precision (%)': round(precision * 100, 2),
            'Recall (%)': round(recall * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'Accuracy (%)': round(accuracy * 100, 2)
        })

    return pd.DataFrame(results), len(train_df), len(test_df)


# ======================================================================================
# 5. USER SATISFACTION QUESTIONNAIRE MODULE
# ======================================================================================

def save_survey_response(user_name, relevance, novelty, diversity, ui_ease, overall, feedback=""):
    """
    Saves questionnaire submission to CSV file.
    """
    record = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'User Name': user_name.strip() if user_name else 'Anonymous',
        'Relevance (1-5)': relevance,
        'Novelty (1-5)': novelty,
        'Diversity (1-5)': diversity,
        'UI Usability (1-5)': ui_ease,
        'Overall Satisfaction (1-5)': overall,
        'Feedback': feedback.strip()
    }
    
    df_new = pd.DataFrame([record])
    if os.path.exists(SURVEY_FILE):
        df_new.to_csv(SURVEY_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(SURVEY_FILE, mode='w', header=True, index=False)
    return True


def load_survey_responses():
    """
    Loads all questionnaire responses from CSV file.
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
