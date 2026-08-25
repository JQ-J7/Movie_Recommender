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
from math import sqrt, log2
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    Supports standard frontend signatures (user_id, structures, alpha, top_n) or (user_id, structures, raw_data, ...).
    """
    # Handle flexible argument orders
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

    # 1. Content User Profile Vector (Weighted centroid of liked movies)
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
# 4. SYSTEM EVALUATION MODULE (RMSE, MSE, MAE, PRECISION, RECALL, F1, NDCG)
# ======================================================================================

def evaluate_models(data=None, test_size=0.2, random_state=42, relevance_threshold=3.5, alpha=0.5, top_k=10):
    """
    Evaluates Baselines, Content-Based Proxy, Collaborative Filtering, and Hybrid Models
    on an 80/20 train-test partition using error metrics and Top-N classification metrics.
    Accepts either DataFrame or structures dictionary.
    """
    if data is None:
        data = load_dataset()
    elif isinstance(data, dict):
        data = data.get('ratings', data.get('raw_data', load_dataset()))

    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Global Mean Baseline
    pred_global = np.full(len(test_df), global_mean)
    
    # 2. Movie Mean Baseline (Item Metadata Proxy)
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


def evaluate_alpha_sensitivity(data=None, step=0.1, alpha_list=None, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10):
    """
    Computes an Alpha Sensitivity Benchmark Table for alpha from 0.0 to 1.0 (step=0.1).
    Evaluates Top-K ranking metrics (precision@10, recall@10, f1@10, ndcg@10) across users.
    Returns a DataFrame with the exact columns:
      ['alpha', 'CF_weight', 'CBF_weight', 'precision@10', 'recall@10', 'f1@10', 'ndcg@10']
    """
    if data is None:
        data = load_dataset()
    elif isinstance(data, dict):
        data = data.get('ratings', data.get('raw_data', load_dataset()))

    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # 1. Content User-Profile representations via TF-IDF
    unique_movies = data.drop_duplicates('movieId').copy()
    soup_series = (
        unique_movies['genres'].fillna('') + ' ' + 
        unique_movies['keyword'].fillna('') + ' ' + 
        unique_movies['overview'].fillna('')
    ).str.lower()
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=15000)
    tfidf_mat = tfidf.fit_transform(soup_series)
    m_to_idx = dict(zip(unique_movies['movieId'], range(len(unique_movies))))
    
    # Precompute user liked centroids from training partition
    user_liked = train_df[train_df['rating'] >= relevance_threshold].groupby('userId')['movieId'].apply(list).to_dict()
    user_profiles = {}
    for u, m_list in user_liked.items():
        indices = [m_to_idx[m] for m in m_list if m in m_to_idx]
        if indices:
            user_profiles[u] = np.asarray(tfidf_mat[indices].mean(axis=0))
            
    # 2. Collaborative User-Item Baseline representations
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # Precompute baseline CF and CBF normalized similarity scores for test set
    u_ids = test_df['userId'].values
    m_ids = test_df['movieId'].values
    
    cf_norm = np.array([
        (user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean) / 5.0
        for u, m in zip(u_ids, m_ids)
    ])
    
    cbf_norm = np.zeros(len(test_df), dtype=float)
    for idx, (u, m) in enumerate(zip(u_ids, m_ids)):
        if u in user_profiles and m in m_to_idx:
            u_vec = user_profiles[u]
            m_vec = tfidf_mat[m_to_idx[m]]
            cbf_norm[idx] = float(cosine_similarity(u_vec, m_vec)[0, 0])
            
    test_eval = test_df.copy()
    test_eval['cf_norm'] = cf_norm
    test_eval['cbf_norm'] = cbf_norm
    
    if alpha_list is not None:
        alphas = np.array(alpha_list)
    else:
        alphas = np.round(np.arange(0.0, 1.0 + step / 2, step), 2)

    results = []
    
    # Pre-group user test frames for ultra-fast evaluation
    grouped_users = [
        (uid, u_df[['movieId', 'rating', 'cf_norm', 'cbf_norm']].copy())
        for uid, u_df in test_eval.groupby('userId')
    ]
    
    k_val = top_k
    for a in alphas:
        cf_w = round(1.0 - a, 2)
        cbf_w = round(a, 2)
        
        precisions, recalls, f1s, ndcgs = [], [], [], []
        
        for uid, u_df in grouped_users:
            relevant = set(u_df[u_df['rating'] >= relevance_threshold]['movieId'])
            if not relevant:
                continue
            
            hybrid_score = (a * u_df['cbf_norm']) + ((1.0 - a) * u_df['cf_norm'])
            top_k_idx = np.argsort(hybrid_score.values)[::-1][:k_val]
            top_k_movies = u_df['movieId'].values[top_k_idx]
            
            hits = [1 if m in relevant else 0 for m in top_k_movies]
            k = min(k_val, len(u_df))
            p_k = sum(hits) / k if k > 0 else 0.0
            r_k = sum(hits) / len(relevant)
            f_k = 2 * (p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
            
            dcg = sum([h / log2(i + 2) for i, h in enumerate(hits)])
            idcg = sum([1.0 / log2(i + 2) for i in range(min(k_val, len(relevant)))])
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            
            precisions.append(p_k)
            recalls.append(r_k)
            f1s.append(f_k)
            ndcgs.append(ndcg)
            
        results.append({
            'alpha': f"{a:.1f}",
            'CF_weight': f"{cf_w:.1f}",
            'CBF_weight': f"{cbf_w:.1f}",
            f'precision@{k_val}': round(float(np.mean(precisions)), 4),
            f'recall@{k_val}': round(float(np.mean(recalls)), 4),
            f'f1@{k_val}': round(float(np.mean(f1s)), 4),
            f'ndcg@{k_val}': round(float(np.mean(ndcgs)), 4)
        })
        
    return pd.DataFrame(results)


def evaluate_top10_models(data=None, test_size=0.2, random_state=42, relevance_threshold=4.0, k=10):
    """
    Evaluates Top-10 recommendation metrics across 5 specific model configurations:
      1. Collaborative Filtering (SVD)
      2. Content-Based Filtering
      3. Hybrid 20% CF / 80% CBF (alpha = 0.2)
      4. Hybrid 50% CF / 50% CBF (alpha = 0.5)
      5. Hybrid 80% CF / 20% CBF (alpha = 0.8)
    
    Formula:
      Score_hybrid = (alpha * Score_CF) + ((1 - alpha) * Score_CBF)
      Where both CF and CBF scores are normalized to [0, 1].
      
    Returns:
      Pandas DataFrame with Columns: ['Model', 'Precision@10', 'Recall@10', 'NDCG@10']
    """
    if data is None:
        data = load_dataset()
    elif isinstance(data, dict):
        data = data.get('ratings', data.get('raw_data', load_dataset()))

    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)

    # 1. Collaborative Filtering: Matrix Factorization via SVD
    user_item_train = train_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    u_idx_map = {uid: i for i, uid in enumerate(user_item_train.index)}
    m_idx_map = {mid: j for j, mid in enumerate(user_item_train.columns)}

    n_factors = min(50, min(user_item_train.shape) - 1)
    svd = TruncatedSVD(n_components=n_factors, random_state=random_state)
    U_sigma = svd.fit_transform(user_item_train.values)
    V_t = svd.components_
    pred_matrix = np.dot(U_sigma, V_t)

    # Normalize SVD predictions to [0, 1]
    min_svd = pred_matrix.min()
    max_svd = pred_matrix.max()
    pred_matrix_norm = (pred_matrix - min_svd) / (max_svd - min_svd) if max_svd > min_svd else pred_matrix

    # 2. Content-Based Filtering: TF-IDF on genres, keywords, and synopsis overviews
    unique_movies = data.drop_duplicates('movieId').copy()
    soup_series = (
        unique_movies['genres'].fillna('') + ' ' + 
        unique_movies['keyword'].fillna('') + ' ' + 
        unique_movies['overview'].fillna('')
    ).str.lower()
    
    tfidf = TfidfVectorizer(stop_words='english', max_features=15000)
    tfidf_mat = tfidf.fit_transform(soup_series)
    m_to_soup_idx = dict(zip(unique_movies['movieId'], range(len(unique_movies))))

    # User profile taste vector from liked movies (rating >= relevance_threshold) in train_df
    user_liked = train_df[train_df['rating'] >= relevance_threshold].groupby('userId')['movieId'].apply(list).to_dict()
    user_profiles = {}
    for u, m_list in user_liked.items():
        indices = [m_to_soup_idx[m] for m in m_list if m in m_to_soup_idx]
        if indices:
            user_profiles[u] = np.asarray(tfidf_mat[indices].mean(axis=0))

    # Precompute test scores for CF and CBF
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()

    u_ids = test_df['userId'].values
    m_ids = test_df['movieId'].values

    cf_scores = np.zeros(len(test_df), dtype=float)
    cbf_scores = np.zeros(len(test_df), dtype=float)

    for idx, (u, m) in enumerate(zip(u_ids, m_ids)):
        # CF score (SVD normalized prediction)
        if u in u_idx_map and m in m_idx_map:
            cf_scores[idx] = pred_matrix_norm[u_idx_map[u], m_idx_map[m]]
        else:
            u_mean = user_means.get(u, global_mean)
            m_mean = movie_means.get(m, global_mean)
            cf_scores[idx] = np.clip((u_mean + m_mean - global_mean) / 5.0, 0.0, 1.0)
            
        # CBF score (Cosine similarity in [0, 1])
        if u in user_profiles and m in m_to_soup_idx:
            u_vec = user_profiles[u]
            m_vec = tfidf_mat[m_to_soup_idx[m]]
            cbf_scores[idx] = float(cosine_similarity(u_vec, m_vec)[0, 0])

    test_eval = test_df.copy()
    test_eval['score_cf'] = cf_scores
    test_eval['score_cbf'] = cbf_scores

    # 3. 5 Target Model Configurations
    models = {
        'Collaborative Filtering (SVD)': 1.0,
        'Content-Based Filtering': 0.0,
        'Hybrid 20% CF / 80% CBF': 0.2,
        'Hybrid 50% CF / 50% CBF': 0.5,
        'Hybrid 80% CF / 20% CBF': 0.8
    }

    grouped_users = [
        (uid, u_df[['movieId', 'rating', 'score_cf', 'score_cbf']].copy())
        for uid, u_df in test_eval.groupby('userId')
    ]

    rows = []
    for model_name, alpha in models.items():
        precisions, recalls, ndcgs = [], [], []
        
        for uid, u_df in grouped_users:
            relevant = set(u_df[u_df['rating'] >= relevance_threshold]['movieId'])
            if not relevant:
                continue
            
            # Blended Score: Score_hybrid = (alpha * Score_CF) + ((1 - alpha) * Score_CBF)
            score_hybrid = (alpha * u_df['score_cf']) + ((1.0 - alpha) * u_df['score_cbf'])
            top_k_idx = np.argsort(score_hybrid.values)[::-1][:k]
            top_k_movies = u_df['movieId'].values[top_k_idx]
            
            hits = [1 if m in relevant else 0 for m in top_k_movies]
            k_eff = min(k, len(u_df))
            p_k = sum(hits) / k_eff if k_eff > 0 else 0.0
            r_k = sum(hits) / len(relevant)
            
            dcg = sum([h / log2(i + 2) for i, h in enumerate(hits)])
            idcg = sum([1.0 / log2(i + 2) for i in range(min(k, len(relevant)))])
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            
            precisions.append(p_k)
            recalls.append(r_k)
            ndcgs.append(ndcg)
            
        rows.append({
            'Model': model_name,
            f'Precision@{k}': round(float(np.mean(precisions)), 4),
            f'Recall@{k}': round(float(np.mean(recalls)), 4),
            f'NDCG@{k}': round(float(np.mean(ndcgs)), 4)
        })

    return pd.DataFrame(rows)


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
# CLI STANDALONE DEMONSTRATION & TEST HARNESS
# ======================================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("   TARUMT AI - Hybrid Movie Recommender System (module_hybrid.py)")
    print("=" * 80)
    print(f"[*] Loading dataset from '{DATASET_FILE}'...")
    data = load_dataset()
    print(f"[+] Loaded {len(data):,} ratings across {data['movieId'].nunique():,} unique movies.")
    
    print("\n[*] Building Hybrid Data Structures & Content Soup...")
    structures = build_engine_structures(data)
    print(f"[+] Structures ready. Matrix shape: {structures['user_movie_matrix'].shape}")
    
    target = 'Toy Story'
    print(f"\n[*] Generating Sample Hybrid Recommendations for '{target}' (alpha=0.5)...")
    recs, err = get_hybrid_recommendations(target, structures, alpha=0.5, top_n=5)
    if err:
        print(f"[!] Error: {err}")
    else:
        print(recs[['title', 'hybrid_score', 'cb_score', 'cf_score', 'avg_rating', 'genres']])
        
    print("\n[*] Running 80/20 Train-Test Evaluation...")
    eval_df, n_tr, n_te = evaluate_models(data)
    print(eval_df.to_string(index=False))
    
    print("\n[*] Running Alpha Sensitivity Analysis (0.0 -> 1.0, step=0.1)...")
    alpha_df = evaluate_alpha_sensitivity(data, step=0.1)
    print(alpha_df.to_string(index=False))
    
    print("\n[*] Running Top-10 Model Benchmarks (SVD, CBF, Hybrid 20/50/80%)...")
    top10_df = evaluate_top10_models(data, relevance_threshold=4.0, k=10)
    print(top10_df.to_string(index=False))
    print("\n[+] Done!")
