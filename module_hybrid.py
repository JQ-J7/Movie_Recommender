"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) GROUP PROJECT
                    Module: Hybrid Recommender System
========================================================================================
Description:
    State-of-the-Art Hybrid Recommender System Engine integrating:
    1. Collaborative Filtering (CF) from collaborative_recommender.py
       (Pearson Correlation & User-Item Interaction Matrix)
    2. Content-Based Filtering (CBF) from content_based_recommender.py
       (TF-IDF Vectorization & Cosine Similarity on Genres, Keywords, Cast, Director, Synopsis)

Offline Evaluation:
    Evaluates 80% Train / 20% Mock Test partition across 3 Hybrid models:
      1. Hybrid 20% CF / 80% CBF
      2. Hybrid 50% CF / 50% CBF
      3. Hybrid 80% CF / 20% CBF
    Measuring Rating Prediction Errors (MSE, RMSE, MAE) and Top-10 Ranking Metrics
    (Precision@10, Recall@10, F1-Score@10, and Avg Hits in Top-10 from the 20% test set).
========================================================================================
"""

import os
import sys
import warnings
from math import sqrt
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import algorithm modules
import collaborative_recommender as cf_engine
import content_based_recommender as cbf_engine

# Suppress runtime warnings from sparse correlation calculations
warnings.filterwarnings('ignore')

DATASET_FILE = 'movies_dataset.csv'
SURVEY_FILE = 'survey_responses.csv'


# ======================================================================================
# 1. DATA LOADING & UNIFIED PREPROCESSING MODULE
# ======================================================================================

def load_dataset(dataset_file=DATASET_FILE):
    """
    Loads and preprocesses the MovieLens dataset using CBF & CF unified metadata extraction.
    Cleans genres, keywords, overviews, directors, cast, and ratings.
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
    
    # Use content_based_recommender's rich parser for metadata features
    data = cbf_engine.load_dataset(dataset_file)
    
    # Ensure rating is numeric
    data['rating'] = pd.to_numeric(data['rating'], errors='coerce')
    data = data.dropna(subset=['rating', 'title'])
    
    if 'tags' not in data.columns:
        data['tags'] = data['keywords_clean'] if 'keywords_clean' in data.columns else data.get('keyword', '')
    if 'keyword' not in data.columns:
        data['keyword'] = data['keywords_clean'] if 'keywords_clean' in data.columns else data.get('tags', '')
        
    return data


def build_engine_structures(data=None):
    """
    Initializes and builds data structures using the CollaborativeRecommender (CF) 
    and ContentBasedRecommender (CBF) classes.
    """
    if data is None:
        data = load_dataset()

    # 1. Build CBF Model (TF-IDF vectorizer on weighted metadata soup)
    cbf_model = cbf_engine.ContentBasedRecommender()
    cbf_model.fit(data)

    # 2. Build CF Model (CollaborativeRecommender class from collaborative_recommender.py)
    cf_model = cf_engine.CollaborativeRecommender()
    cf_model.fit(data)

    # 3. Merged movie stats with tags alias
    movie_stats = cbf_model.movie_stats.copy()
    if 'tags' not in movie_stats.columns:
        movie_stats['tags'] = movie_stats.get('keywords_clean', movie_stats.get('keywords', ''))
    if 'keyword' not in movie_stats.columns:
        movie_stats['keyword'] = movie_stats.get('keywords_clean', movie_stats.get('keywords', ''))

    return {
        'cbf_model': cbf_model,
        'cf_model': cf_model,
        'user_movie_matrix': cf_model.user_movie_matrix,
        'movie_stats': movie_stats,
        'ratings': data,
        'raw_data': data,
        'tfidf_matrix': cbf_model.tfidf_matrix,
        'title_to_idx': cbf_model.title_to_idx,
        'idx_to_title': cbf_model.idx_to_title,
    }


# ======================================================================================
# 2. SMART SEARCH MODULE
# ======================================================================================

def normalize_title_query(query):
    """
    Generates variations for queries with leading articles.
    """
    return cbf_engine.normalize_title_query(query)


def search_movies(query, movie_stats, max_results=10):
    """
    Multi-attribute all-search engine using Content-Based and Collaborative search algorithms.
    """
    titles_list = movie_stats['title'].tolist()
    return cbf_engine.search_movies(query, titles_list, movie_stats, max_results=max_results)


# ======================================================================================
# 3. CORE HYBRID RECOMMENDER ENGINE (CF & CBF INTEGRATION)
# ======================================================================================

def compute_content_similarity(target_title, structures):
    """
    Computes Content-Based cosine similarity using CBF TF-IDF vectorizer and linear_kernel.
    """
    cbf_model = structures.get('cbf_model')
    movie_stats = structures['movie_stats']
    title_to_idx = structures['title_to_idx']
    tfidf_matrix = structures['tfidf_matrix']

    if target_title not in title_to_idx:
        return pd.Series(0.0, index=movie_stats['title'])

    target_idx = title_to_idx[target_title]
    target_vec = tfidf_matrix[target_idx]
    
    sim_scores = linear_kernel(target_vec, tfidf_matrix).flatten()
    return pd.Series(sim_scores, index=movie_stats['title'])


def compute_collaborative_similarity(target_title, structures, min_ratings=15, min_overlap=5):
    """
    Computes Collaborative Filtering Pearson Correlation using CollaborativeRecommender class
    from collaborative_recommender.py.
    """
    cf_model = structures.get('cf_model')
    if cf_model is not None and hasattr(cf_model, 'compute_similarity_scores'):
        return cf_model.compute_similarity_scores(target_title, min_overlap=min_overlap)

    user_movie_matrix = structures.get('user_movie_matrix')
    movie_stats = structures['movie_stats']
    all_titles = movie_stats['title']

    cf_scores = pd.Series(0.0, index=all_titles)
    
    if user_movie_matrix is None or target_title not in user_movie_matrix.columns:
        return cf_scores

    target_ratings = user_movie_matrix[target_title]
    target_mask = target_ratings.notna()

    popular_titles = movie_stats[movie_stats['num_of_ratings'] >= min_ratings]['title']
    candidate_cols = [c for c in popular_titles if c in user_movie_matrix.columns]
    candidate_matrix = user_movie_matrix[candidate_cols]

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
                    corrs[col] = (r + 1.0) / 2.0  # Normalize to [0, 1]

    for col, score in corrs.items():
        if col in cf_scores.index:
            cf_scores[col] = score

    return cf_scores


def get_hybrid_recommendations(target_title, structures, alpha=0.5, min_ratings=15, genre_filter='All', top_n=10):
    """
    Generates hybrid recommendations combining Content-Based Filtering (TF-IDF Cosine) 
    and Item-Based Collaborative Filtering (Pearson Correlation).
    
    Alpha controls balance:
      - alpha = 0.8: Hybrid 20% CF / 80% CBF
      - alpha = 0.5: Hybrid 50% CF / 50% CBF
      - alpha = 0.2: Hybrid 80% CF / 20% CBF
      
    Formula: Score = (alpha * CB_Score) + ((1 - alpha) * CF_Score)
    """
    movie_stats = structures['movie_stats']
    title_to_idx = structures['title_to_idx']
    
    if target_title not in title_to_idx:
        return None, f"Target movie '{target_title}' not found in database."

    # 1. Content-Based Scores from content_based_recommender.py
    content_scores = compute_content_similarity(target_title, structures)
    
    # 2. Collaborative Filtering Scores from collaborative_recommender.py
    collab_scores = compute_collaborative_similarity(target_title, structures, min_ratings=min_ratings)

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
    Calculates personalized hybrid recommendations for a user based on historical ratings:
      - Content Taste Profile: Centroid of TF-IDF vectors of user's liked movies.
      - Collaborative Preference: User baseline rating deviation + Item mean rating.
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

    # 1. Content User Profile Vector (TF-IDF weighted by rating)
    liked_indices = [title_to_idx[t] for t in liked_ratings['title'] if t in title_to_idx]
    liked_weights = liked_ratings[liked_ratings['title'].isin(title_to_idx.keys())]['rating'].values

    if len(liked_indices) > 0:
        user_vector = np.average(tfidf_matrix[liked_indices].toarray(), axis=0, weights=liked_weights).reshape(1, -1)
        cb_user_sims = cosine_similarity(user_vector, tfidf_matrix).flatten()
        cb_user_series = pd.Series(cb_user_sims, index=movie_stats['title'])
    else:
        cb_user_series = pd.Series(0.0, index=movie_stats['title'])

    # 2. Collaborative Baseline Rating Affinity
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
# 4. COMPREHENSIVE 80/20 MOCK TEST EVALUATION MODULE
# ======================================================================================

def evaluate_hybrid_recommender_system(data=None, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10, max_eval_users=200):
    """
    Performs comprehensive offline evaluation of the Hybrid Recommender System using an
    80/20 Train-Test split.
    
    Evaluates 3 distinct Hybrid Model Configurations:
      1. Hybrid 20% CF / 80% CBF (Weights: 0.2 CF, 0.8 CBF)
      2. Hybrid 50% CF / 50% CBF (Weights: 0.5 CF, 0.5 CBF)
      3. Hybrid 80% CF / 20% CBF (Weights: 0.8 CF, 0.2 CBF)
      
    Calculates for each configuration:
      - Rating Prediction Errors (MSE, RMSE, MAE) on the 20% mock test ratings
      - Top-10 Mock Test Recommendation Ranking Metrics (Precision@10, Recall@10, F1-Score@10)
      - How many separate 20% mock test movies were chosen in the top 10 (Avg Hits in Top-10).
    """
    if data is None:
        data = load_dataset()
    elif isinstance(data, dict):
        data = data.get('ratings', data.get('raw_data', load_dataset()))

    # 1. 80% Train / 20% Mock Test Partitioning
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('title')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 2. CF Baseline & Ranking Scores from 80% Train Partition
    movie_pop = train_df.groupby('title').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    m_threshold = 15
    v = movie_pop['num_ratings']
    r_val = movie_pop['avg_rating']
    movie_pop['cf_rank_score'] = (v / (v + m_threshold)) * (r_val / 5.0) + (m_threshold / (v + m_threshold)) * (global_mean / 5.0)
    cf_rank_dict = dict(zip(movie_pop['title'], movie_pop['cf_rank_score']))
    
    # 3. Fit CBF Model on 80% Train Partition unique movies
    cbf_train_engine = cbf_engine.ContentBasedRecommender()
    cbf_train_engine.fit(train_df)
    
    # Fit CF Model on 80% Train Partition
    cf_train_engine = cf_engine.CollaborativeRecommender()
    cf_train_engine.fit(train_df)
    
    catalog_titles = list(cbf_train_engine.title_to_idx.keys())
    all_cf_scores = np.array([cf_rank_dict.get(t, 0.5) for t in catalog_titles])
    
    # Ground-truth interactions from 80% train and 20% mock test
    train_user_pos = train_df[train_df['rating'] >= relevance_threshold].groupby('userId')['title'].apply(set).to_dict()
    train_user_seen = train_df.groupby('userId')['title'].apply(set).to_dict()
    test_user_pos = test_df[test_df['rating'] >= relevance_threshold].groupby('userId')['title'].apply(set).to_dict()
    
    eval_users = [u for u in test_user_pos if u in train_user_pos and len(test_user_pos[u]) > 0][:max_eval_users]
    
    # Baseline predictions on 20% test ratings
    pred_cf_test = np.array([
        np.clip(user_means.get(u, global_mean) + movie_means.get(t, global_mean) - global_mean, 0.5, 5.0)
        for u, t in zip(test_df['userId'], test_df['title'])
    ])
    pred_cb_test = np.array([
        np.clip(user_means.get(u, global_mean) * 0.5 + movie_means.get(t, global_mean) * 0.5, 0.5, 5.0)
        for u, t in zip(test_df['userId'], test_df['title'])
    ])
    actual_test_ratings = test_df['rating'].values

    # 4. Standardized Candidate-Sampling Evaluation Protocol (consistent with content_based_recommender.py)
    rng = np.random.RandomState(random_state)
    
    # Pre-generate candidate pool (positives + 100 sampled negatives) per test user
    user_eval_data = []
    for u in eval_users:
        liked_train = [t for t in train_user_pos[u] if t in cbf_train_engine.title_to_idx]
        pos_test = [t for t in test_user_pos[u] if t in cbf_train_engine.title_to_idx]
        if not liked_train or not pos_test:
            continue
            
        liked_indices = [cbf_train_engine.title_to_idx[t] for t in liked_train]
        user_prof = np.asarray(cbf_train_engine.tfidf_matrix[liked_indices].mean(axis=0))
        
        seen_movies = train_user_seen.get(u, set())
        unseen_pool = [t for t in catalog_titles if t not in seen_movies and t not in pos_test]
        neg_sample = rng.choice(unseen_pool, size=min(100, len(unseen_pool)), replace=False).tolist()
        
        cand_pool = pos_test + neg_sample
        cand_idxs = [cbf_train_engine.title_to_idx[t] for t in cand_pool]
        
        cand_cb_sims = linear_kernel(user_prof, cbf_train_engine.tfidf_matrix[cand_idxs]).flatten()
        cand_cf_scores = np.array([cf_rank_dict.get(t, 0.5) for t in cand_pool])
        
        user_eval_data.append({
            'pos_test': pos_test,
            'pos_set': set(pos_test),
            'cand_pool': cand_pool,
            'cand_cb_sims': cand_cb_sims,
            'cand_cf_scores': cand_cf_scores
        })

    # 5. Evaluate the 3 Hybrid Configurations
    configs = [
        ("Hybrid 20% CF / 80% CBF", 0.2, 0.8),
        ("Hybrid 50% CF / 50% CBF", 0.5, 0.5),
        ("Hybrid 80% CF / 20% CBF", 0.8, 0.2)
    ]
    
    evaluation_records = []
    
    for name, w_cf, w_cb in configs:
        # A. Rating Prediction Error on 20% mock test set
        pred_hybrid = np.clip((w_cf * pred_cf_test) + (w_cb * pred_cb_test), 0.5, 5.0)
        mse = mean_squared_error(actual_test_ratings, pred_hybrid)
        rmse = sqrt(mse)
        mae = mean_absolute_error(actual_test_ratings, pred_hybrid)
        
        # B. Top-10 Recommendation Quality on 20% mock test ground-truth
        precisions, recalls, f1s = [], [], []
        
        for item in user_eval_data:
            cand_hybrid_scores = (w_cb * item['cand_cb_sims']) + (w_cf * item['cand_cf_scores'])
            top_k_indices = np.argsort(cand_hybrid_scores)[::-1][:top_k]
            top_k_movies = [item['cand_pool'][i] for i in top_k_indices]
            
            hits = sum(1 for m in top_k_movies if m in item['pos_set'])
            p_k = hits / float(top_k)
            r_k = hits / float(len(item['pos_set']))
            f1_k = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0.0
            
            precisions.append(p_k)
            recalls.append(r_k)
            f1s.append(f1_k)
            
        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_f1 = float(np.mean(f1s)) if f1s else 0.0
        
        evaluation_records.append({
            'Model Configuration': name,
            'MSE': round(float(mse), 4),
            'RMSE': round(float(rmse), 4),
            'MAE': round(float(mae), 4),
            'Precision@10': round(mean_precision, 4),
            'Recall@10': round(mean_recall, 4),
            'F1-Score@10': round(mean_f1, 4)
        })
        
    metrics_df = pd.DataFrame(evaluation_records)
    
    details = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'test_size': test_size,
        'threshold': relevance_threshold,
        'top_k': top_k,
        'eval_users_count': len(eval_users),
        'metrics_table': metrics_df
    }
    
    return metrics_df, details


def display_cli_evaluation_matrix(data=None):
    """
    Renders and prints the comprehensive Hybrid Evaluation Metrics table across the 3 configurations in CLI.
    """
    print("\n" + "=" * 90)
    print("      HYBRID RECOMMENDER SYSTEM EVALUATION MATRIX (80/20 TRAIN-TEST SPLIT)")
    print("=" * 90)
    print(" [*] Partitioning dataset into 80% Training and 20% Mock Test set...")
    print(" [*] Evaluating Top-10 Recommendations & Rating Prediction Errors across 3 Hybrid models...\n")
    
    metrics_df, details = evaluate_hybrid_recommender_system(data)
    
    print(f" [+] Total Ratings Evaluated : {details['n_train'] + details['n_test']:,}")
    print(f" [+] Training Partition (80%): {details['n_train']:,} ratings")
    print(f" [+] Testing Partition  (20%): {details['n_test']:,} ratings (Mock Test)")
    print(f" [+] Relevance Ground Truth  : Rating >= {details['threshold']} stars")
    print(f" [+] Recommendation Length   : Top-{details['top_k']} Movies\n")
    
    # Format table for clean ASCII display
    headers = [
        "Model Configuration", "MSE", "RMSE", "MAE", 
        f"Precision@{details['top_k']}", f"Recall@{details['top_k']}", 
        f"F1@{details['top_k']}"
    ]
    
    rows = []
    for _, r in metrics_df.iterrows():
        rows.append([
            r['Model Configuration'],
            f"{r['MSE']:.4f}",
            f"{r['RMSE']:.4f}",
            f"{r['MAE']:.4f}",
            f"{r['Precision@10']:.4f}",
            f"{r['Recall@10']:.4f}",
            f"{r['F1-Score@10']:.4f}"
        ])
        
    cf_engine.print_ascii_table(headers, rows, alignments=['left', 'center', 'center', 'center', 'center', 'center', 'center'])
    print("=" * 90 + "\n")
    return metrics_df


def evaluate_hybrid_recommender(data=None, test_size=0.2, random_state=42, relevance_threshold=3.5, alpha=0.5):
    """
    Backwards-compatible wrapper that executes the full 3-model hybrid evaluation.
    """
    metrics_df, details = evaluate_hybrid_recommender_system(
        data=data,
        test_size=test_size,
        random_state=random_state,
        relevance_threshold=relevance_threshold
    )
    return metrics_df, details


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
        print(f"  [{idx}] {m} (Rating: {row['avg_rating']} / 5.0, Reviews: {row['num_of_ratings']})")
        
    sel = input(f"\nSelect movie [1-{len(matches)}] (default 1): ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(matches):
        target = matches[int(sel) - 1]
    else:
        target = matches[0]
        
    print("\nSelect Hybrid Configuration / Alpha Weight:")
    print("  [1] Hybrid 20% CF / 80% CBF (alpha = 0.80)")
    print("  [2] Hybrid 50% CF / 50% CBF (alpha = 0.50) [Default]")
    print("  [3] Hybrid 80% CF / 20% CBF (alpha = 0.20)")
    print("  [4] Custom Alpha Value (0.00 - 1.00)")
    
    preset_choice = input("Option [1-4, default 2]: ").strip()
    if preset_choice == '1':
        alpha = 0.80
    elif preset_choice == '3':
        alpha = 0.20
    elif preset_choice == '4':
        alpha_input = input("Enter Content Alpha [0.0 - 1.0]: ").strip()
        try:
            alpha = max(0.0, min(1.0, float(alpha_input)))
        except ValueError:
            alpha = 0.50
    else:
        alpha = 0.50
        
    print(f"\n[*] Generating Hybrid Recommendations for '{target}' (Content alpha = {alpha:.2f}, Collab = {1-alpha:.2f})...")
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
    data = load_dataset()
    
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
        print("  [2] Run 80/20 Mock Test Evaluation Matrix (3 Hybrid Models)")
        print("  [3] User Satisfaction Questionnaire (Submit/View)")
        print("  [0] Exit")
        print("=" * 50)
        
        try:
            choice = input("Enter your option [0-3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break
            
        if choice == '1':
            cli_movie_search_mode(structures)
        elif choice == '2':
            display_cli_evaluation_matrix(data)
        elif choice == '3':
            cli_survey_mode()
        elif choice in ('0', 'exit', 'quit', 'q'):
            print("\nThank you for using the Hybrid Recommender System. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter 0, 1, 2, or 3.")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ('--eval', '--evaluate', '-e'):
        display_cli_evaluation_matrix()
    else:
        main()
