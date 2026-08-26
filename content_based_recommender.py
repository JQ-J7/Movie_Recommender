"""
========================================================================================
             TARUMT - ARTIFICIAL INTELLIGENCE (AI) REPOSITORY
         Content-Based Filtering (CBF) Movie Recommender System
========================================================================================
Description:
    Content-Based Movie Recommender Engine using TF-IDF and Cosine Similarity
    on metadata (genres, keywords, cast, director, and plot overview).

Key Features:
    1. Weighted Metadata Vectorization (TF-IDF & Cosine Similarity).
    2. Intelligent Search Engine (Exact, Substring, Fuzzy & Article Normalization).
    3. Top-10 Similar Movie Recommendations with Token-level Explainability.
    4. Comprehensive Offline Evaluation (80/20 Split: MSE, RMSE, Precision@K, Recall@K, F1@K).
    5. Interactive Console Interface & Metadata Analytics.
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
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress runtime warnings
warnings.filterwarnings('ignore')


# ======================================================================================
# 1. METADATA PARSING & DATA PREPROCESSING MODULE
# ======================================================================================

def fast_extract_names(val):
    """
    Extracts 'name' values from JSON-like strings or pipe-delimited strings.
    Example: "[{'id': 18, 'name': 'Drama'}, {'id': 80, 'name': 'Crime'}]" -> "Drama|Crime"
    """
    if not isinstance(val, str) or not val:
        return ''
    if val.startswith('['):
        names = re.findall(r"'name':\s*'([^']*)'", val)
        if not names:
            names = re.findall(r'"name":\s*"([^"]*)"', val)
        if names:
            return '|'.join(names)
    return val


def clean_metadata_token(token):
    """
    Converts multi-word tokens into single entity strings (e.g. 'Science Fiction' -> 'sciencefiction',
    'Christopher Nolan' -> 'christophernolan') and lowercases them to prevent ambiguous unigram splits.
    """
    if not isinstance(token, str):
        return ''
    return re.sub(r'[^a-zA-Z0-9]', '', token).lower()


def parse_delimited_tokens(val, delimiter='|', max_tokens=None):
    """
    Splits delimited strings, cleans each token into unified entities, and returns joined tokens.
    """
    if not isinstance(val, str) or not val.strip():
        return ''
    raw_tokens = [t.strip() for t in val.split(delimiter) if t.strip()]
    if max_tokens:
        raw_tokens = raw_tokens[:max_tokens]
    clean_tokens = [clean_metadata_token(t) for t in raw_tokens if clean_metadata_token(t)]
    return ' '.join(clean_tokens)


def build_metadata_soup(row, genre_weight=3, keyword_weight=2, cast_weight=2, director_weight=3):
    """
    Constructs a weighted 'metadata soup' document for each movie combining:
      - Genres (weighted repetition)
      - Plot Overview / Synopsis (raw cleaned text)
      - Keywords / Tags (weighted repetition)
      - Cast members (if present)
      - Directors / Filmmakers (if present)
    """
    parts = []
    
    # 1. Genres (high semantic importance)
    genres_tokens = parse_delimited_tokens(row.get('genres_clean', ''))
    if genres_tokens:
        parts.extend([genres_tokens] * genre_weight)
        
    # 2. Keywords / Tags
    keywords_tokens = parse_delimited_tokens(row.get('keywords_clean', ''))
    if keywords_tokens:
        parts.extend([keywords_tokens] * keyword_weight)
        
    # 3. Director / Filmmakers (if available)
    if 'director_clean' in row and row['director_clean']:
        director_tokens = parse_delimited_tokens(row['director_clean'])
        if director_tokens:
            parts.extend([director_tokens] * director_weight)
            
    # 4. Top Cast members (if available)
    if 'cast_clean' in row and row['cast_clean']:
        cast_tokens = parse_delimited_tokens(row['cast_clean'], max_tokens=4)
        if cast_tokens:
            parts.extend([cast_tokens] * cast_weight)
            
    # 5. Plot Overview / Synopsis
    overview = str(row.get('overview_clean', '')).strip()
    if overview and overview.lower() != 'nan':
        # Clean overview text
        overview_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', overview).lower()
        parts.append(overview_clean)
        
    soup = ' '.join([p for p in parts if p]).strip()
    return soup if soup else 'movie film story'


def load_dataset(dataset_file='movies_dataset.csv'):
    """
    Loads dataset file and cleans metadata features for Content-Based Filtering.
    """
    if not os.path.exists(dataset_file):
        print(f"[!] Error: Dataset file '{dataset_file}' not found.")
        print(f"    Please ensure '{dataset_file}' is in the current directory.")
        return None
        
    print(f"[+] Loading dataset from '{dataset_file}'...")
    data = pd.read_csv(dataset_file)
    
    # Standardize column naming
    if 'genres' in data.columns:
        data['genres_clean'] = data['genres'].apply(fast_extract_names)
    else:
        data['genres_clean'] = ''
        
    keyword_col = 'keyword' if 'keyword' in data.columns else ('keywords' if 'keywords' in data.columns else ('tags' if 'tags' in data.columns else None))
    if keyword_col:
        data['keywords_clean'] = data[keyword_col].apply(fast_extract_names)
    else:
        data['keywords_clean'] = ''
        
    if 'overview' in data.columns:
        data['overview_clean'] = data['overview'].fillna('')
    else:
        data['overview_clean'] = ''
        
    if 'director' in data.columns:
        data['director_clean'] = data['director'].apply(fast_extract_names)
    elif 'crew' in data.columns:
        # Extract director from crew JSON if present
        def extract_director(crew_str):
            if not isinstance(crew_str, str):
                return ''
            m = re.findall(r"'job':\s*'Director',\s*'name':\s*'([^']*)'", crew_str)
            return '|'.join(m) if m else ''
        data['director_clean'] = data['crew'].apply(extract_director)
    else:
        data['director_clean'] = ''
        
    if 'cast' in data.columns:
        data['cast_clean'] = data['cast'].apply(fast_extract_names)
    else:
        data['cast_clean'] = ''
        
    print(f"[+] Successfully loaded {len(data):,} ratings across {data['title'].nunique():,} unique titles.\n")
    return data


# ======================================================================================
# 2. CONTENT-BASED RECOMMENDER ENGINE (TF-IDF & COSINE SIMILARITY)
# ======================================================================================

class ContentBasedRecommender:
    """
    Content-Based Filtering Recommender Engine using TF-IDF and Cosine Similarity.
    """
    def __init__(self, max_features=25000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.tfidf_matrix = None
        self.movie_stats = None
        self.title_to_idx = {}
        self.idx_to_title = {}
        self.data = None

    def fit(self, data):
        """
        Builds movie catalog aggregation, constructs metadata soups, and fits TF-IDF vectorizer.
        """
        print("[*] Processing Movie Metadata & Aggregating Catalog...")
        self.data = data
        
        # Aggregate unique movie records
        agg_dict = {
            'movieId': ('movieId', 'first') if 'movieId' in data.columns else ('title', 'count'),
            'genres': ('genres_clean', 'first'),
            'keywords': ('keywords_clean', 'first'),
            'overview': ('overview_clean', 'first'),
            'genres_clean': ('genres_clean', 'first'),
            'keywords_clean': ('keywords_clean', 'first'),
            'overview_clean': ('overview_clean', 'first'),
            'director_clean': ('director_clean', 'first'),
            'cast_clean': ('cast_clean', 'first'),
        }
        if 'rating' in data.columns:
            agg_dict['avg_rating'] = ('rating', 'mean')
            agg_dict['num_of_ratings'] = ('rating', 'count')
            
        movie_stats = data.groupby('title').agg(**agg_dict).reset_index()
        
        if 'avg_rating' in movie_stats.columns:
            movie_stats['avg_rating'] = movie_stats['avg_rating'].round(2)
        else:
            movie_stats['avg_rating'] = 0.0
            movie_stats['num_of_ratings'] = 0

        print("[*] Constructing Weighted Metadata Soups (Genres, Keywords, Cast, Directors, Synopsis)...")
        movie_stats['metadata_soup'] = movie_stats.apply(build_metadata_soup, axis=1)

        print("[*] Computing TF-IDF Matrix (Term Frequency-Inverse Document Frequency)...")
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            token_pattern=r'(?u)\b[a-zA-Z0-9_-]+\b'
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(movie_stats['metadata_soup'])
        
        self.movie_stats = movie_stats
        self.title_to_idx = {title: idx for idx, title in enumerate(movie_stats['title'])}
        self.idx_to_title = {idx: title for idx, title in enumerate(movie_stats['title'])}
        
        print(f"[+] TF-IDF Matrix ready: {self.tfidf_matrix.shape[0]:,} movies x {self.tfidf_matrix.shape[1]:,} features.")
        print(f"[+] Sparsity of TF-IDF feature space: {(1.0 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))) * 100:.3f}%\n")
        return self

    def get_similar_movies(self, target_title, top_n=10, min_similarity=0.01):
        """
        Computes Cosine Similarity between the target movie and all other movies in the catalog.
        Enforces a maximum limit of 10 recommendations.
        
        Returns:
            pd.DataFrame: Top similar movies with similarity scores and metadata.
        """
        if target_title not in self.title_to_idx:
            return None
            
        # Enforce maximum limit of 10 movies
        top_n = min(max(1, int(top_n)), 10)
            
        target_idx = self.title_to_idx[target_title]
        target_vector = self.tfidf_matrix[target_idx]
        
        # Compute Cosine Similarity via fast linear_kernel (since TF-IDF vectors are L2-normalized)
        sim_scores = linear_kernel(target_vector, self.tfidf_matrix).flatten()
        
        # Pair indices with similarity scores
        sim_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in sim_indices:
            if idx == target_idx:
                continue  # Skip target movie itself
            score = sim_scores[idx]
            if score < min_similarity:
                break
                
            row = self.movie_stats.iloc[idx]
            results.append({
                'title': row['title'],
                'Similarity': round(float(score), 4),
                'avg_rating': row['avg_rating'],
                'num_of_ratings': row['num_of_ratings'],
                'genres': row['genres_clean'] if row['genres_clean'] else 'N/A',
                'keywords': row['keywords_clean'][:60] + '...' if len(str(row['keywords_clean'])) > 60 else (row['keywords_clean'] or 'N/A'),
                'overview': row['overview_clean'][:90] + '...' if len(str(row['overview_clean'])) > 90 else (row['overview_clean'] or 'N/A')
            })
            if len(results) >= top_n:
                break
                
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results)

    def explain_similarity(self, target_title, recommended_title, top_k_features=5):
        """
        Explains why a recommendation was generated by extracting the highest contributing TF-IDF tokens.
        """
        if target_title not in self.title_to_idx or recommended_title not in self.title_to_idx:
            return []
            
        idx_a = self.title_to_idx[target_title]
        idx_b = self.title_to_idx[recommended_title]
        
        vec_a = self.tfidf_matrix[idx_a].toarray().flatten()
        vec_b = self.tfidf_matrix[idx_b].toarray().flatten()
        
        # Element-wise product gives exact contribution to dot product (cosine similarity)
        contrib = vec_a * vec_b
        top_feature_indices = np.argsort(contrib)[::-1][:top_k_features]
        
        feature_names = self.vectorizer.get_feature_names_out()
        explanations = []
        for fi in top_feature_indices:
            if contrib[fi] > 0:
                explanations.append((feature_names[fi], round(float(contrib[fi]), 4)))
        return explanations

    def recommend_for_user_profile(self, user_id, top_n=10, min_rating=3.0):
        """
        Generates Content-Based recommendations personalized for a specific user.
        Builds a User Preference Vector by aggregating TF-IDF vectors of movies the user liked/rated,
        weighted by their rating deviations from the user's mean.
        """
        if self.data is None or 'userId' not in self.data.columns:
            print("[!] User rating interaction history is required for user profile recommendations.")
            return None
            
        user_ratings = self.data[self.data['userId'] == user_id]
        if user_ratings.empty:
            print(f"[!] No rating history found for User ID {user_id}.")
            return None
            
        # Filter movies the user has positively rated
        positive_ratings = user_ratings[user_ratings['rating'] >= min_rating]
        if positive_ratings.empty:
            positive_ratings = user_ratings
            
        user_mean = user_ratings['rating'].mean()
        weighted_vector = np.zeros((1, self.tfidf_matrix.shape[1]))
        rated_indices = set()
        
        for _, row in positive_ratings.iterrows():
            title = row['title']
            if title in self.title_to_idx:
                idx = self.title_to_idx[title]
                rated_indices.add(idx)
                # Weight = (rating - user_mean + 1.0)
                weight = max(0.1, float(row['rating']) - user_mean + 1.0)
                weighted_vector += weight * self.tfidf_matrix[idx].toarray()
                
        # Normalize user vector
        norm = np.linalg.norm(weighted_vector)
        if norm > 0:
            weighted_vector /= norm
            
        sim_scores = cosine_similarity(weighted_vector, self.tfidf_matrix).flatten()
        sim_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in sim_indices:
            if idx in rated_indices:
                continue  # Don't recommend already seen/rated movies
            score = sim_scores[idx]
            row = self.movie_stats.iloc[idx]
            results.append({
                'title': row['title'],
                'Profile Match': round(float(score), 4),
                'avg_rating': row['avg_rating'],
                'num_of_ratings': row['num_of_ratings'],
                'genres': row['genres_clean'] if row['genres_clean'] else 'N/A',
                'keywords': row['keywords_clean'][:60] + '...' if len(str(row['keywords_clean'])) > 60 else (row['keywords_clean'] or 'N/A')
            })
            if len(results) >= top_n:
                break
                
        return pd.DataFrame(results)

    def recommend_for_cold_start_item(self, new_title, genres='', keywords='', overview='', director='', cast='', top_n=10):
        """
        Demonstrates Content-Based Filtering's Cold-Start capability:
        Generates recommendations for an arbitrary brand-new movie with 0 historical ratings
        solely from its textual metadata.
        """
        temp_row = {
            'genres_clean': fast_extract_names(genres) if '[' in genres else genres,
            'keywords_clean': fast_extract_names(keywords) if '[' in keywords else keywords,
            'overview_clean': overview,
            'director_clean': director,
            'cast_clean': cast
        }
        soup = build_metadata_soup(temp_row)
        new_vec = self.vectorizer.transform([soup])
        sim_scores = linear_kernel(new_vec, self.tfidf_matrix).flatten()
        sim_indices = np.argsort(sim_scores)[::-1]
        
        results = []
        for idx in sim_indices:
            score = sim_scores[idx]
            row = self.movie_stats.iloc[idx]
            if row['title'].lower() == new_title.lower():
                continue
            results.append({
                'title': row['title'],
                'Similarity': round(float(score), 4),
                'avg_rating': row['avg_rating'],
                'num_of_ratings': row['num_of_ratings'],
                'genres': row['genres_clean'] if row['genres_clean'] else 'N/A',
                'keywords': row['keywords_clean'][:60] + '...' if len(str(row['keywords_clean'])) > 60 else (row['keywords_clean'] or 'N/A')
            })
            if len(results) >= top_n:
                break
        return pd.DataFrame(results)


# ======================================================================================
# 3. INTELLIGENT SEARCH ENGINE MODULE
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
    1. Exact case-insensitive match on Title.
    2. Candidate scoring across Titles, Keywords/Tags, Genres, and Plot Overview.
    3. Fuzzy string matching fallback.
    """
    query_clean = query.strip()
    query_lower = query_clean.lower()
    query_variants = normalize_title_query(query_clean)
    stats_map = dict(zip(movie_stats['title'], movie_stats['num_of_ratings']))
    
    # 2. Scored multi-attribute candidates
    scored_candidates = {}
    
    # Exact Title Match (Highest Weight: 5000 + popularity)
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
            if title.lower() == var_lower or clean_title == var_lower:
                pop = stats_map.get(title, 0)
                scored_candidates[title] = max(scored_candidates.get(title, 0), 5000 + pop)
                
    # Title Substring (Weight: 1000 + popularity)
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if var_lower in title.lower():
                pop = stats_map.get(title, 0)
                scored_candidates[title] = max(scored_candidates.get(title, 0), 1000 + pop)
                
    # Keywords / Tags (Weight: 500 + popularity)
    if 'keywords_clean' in movie_stats.columns:
        kw_matches = movie_stats[movie_stats['keywords_clean'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in kw_matches.iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 500 + pop)
            
    # Genres (Weight: 200 + popularity)
    if 'genres_clean' in movie_stats.columns:
        genre_matches = movie_stats[movie_stats['genres_clean'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in genre_matches.iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 200 + pop)
            
    # Overview / Synopsis matches (Weight: 150 + popularity)
    if 'overview_clean' in movie_stats.columns:
        ov_matches = movie_stats[movie_stats['overview_clean'].str.contains(r'(?i)\b' + re.escape(query_clean), na=False, regex=True)]
        if ov_matches.empty:
            ov_matches = movie_stats[movie_stats['overview_clean'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in ov_matches.head(15).iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 150 + pop)
            
    if scored_candidates:
        sorted_candidates = sorted(scored_candidates.keys(), key=lambda t: scored_candidates[t], reverse=True)
        return sorted_candidates[:max_results]
        
    # 3. Fuzzy matching fallback
    fuzzy_matches = difflib.get_close_matches(query_clean, titles_list, n=max_results, cutoff=0.4)
    return fuzzy_matches


# ======================================================================================
# 4. TABULAR FORMATTING & EVALUATION MODULES
# ======================================================================================

def print_ascii_table(data, max_col_widths=None):
    """
    Renders a clean ASCII boxed table with vertical column dividers '|' and horizontal borders '+---+'.
    Accepts a pandas DataFrame, list of dicts, or list of lists.
    """
    if isinstance(data, pd.DataFrame):
        headers = list(data.columns)
        rows = [[str(val) for val in row] for row in data.values]
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        headers = list(data[0].keys())
        rows = [[str(d.get(h, '')) for h in headers] for d in data]
    else:
        rows = [[str(val) for val in row] for row in (data or [])]
        headers = [f"Col {i+1}" for i in range(len(rows[0]))] if rows else []

    if not rows:
        print("+--------------------------+")
        print("| No records to display    |")
        print("+--------------------------+")
        return

    # Calculate initial column widths based on headers and cell contents
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(str(val)))

    # Apply maximum column width constraints if provided
    if max_col_widths:
        for idx, max_w in max_col_widths.items():
            col_idx = headers.index(idx) if isinstance(idx, str) and idx in headers else (idx if isinstance(idx, int) else None)
            if col_idx is not None and col_idx < len(col_widths):
                col_widths[col_idx] = min(col_widths[col_idx], max_w)

    # Build horizontal divider line
    sep_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"

    # Build formatted header row
    header_row = "| " + " | ".join([f"{str(h):<{w}}" for h, w in zip(headers, col_widths)]) + " |"

    print(sep_line)
    print(header_row)
    print(sep_line)

    # Print data rows with text truncation if exceeding width
    for row in rows:
        formatted_cells = []
        for idx, (val, w) in enumerate(zip(row, col_widths)):
            val_str = str(val)
            if len(val_str) > w:
                val_str = val_str[:max(0, w - 3)] + "..."
            formatted_cells.append(f"{val_str:<{w}}")
        print("| " + " | ".join(formatted_cells) + " |")

    print(sep_line)


def evaluate_content_based_system(cbf_model, data, test_size=0.2, random_state=42, relevance_threshold=3.5, k=10, sample_users=200):
    """
    Conducts comprehensive offline evaluation of the Content-Based Filtering Recommender:
    - Step 1: Data split (80/20)
    - Step 2: Training CBF Model on Train set
    - Step 3: Evaluating Predictions (MSE, RMSE)
    - Step 4: Evaluating Ranking & Classification Metrics (Precision@K, Recall@K, F1@K)
    """
    print("\nLoading dataset for evaluation...")
    print(f"Splitting data into Train and Test sets ({int((1-test_size)*100)}/{int(test_size*100)})...")
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    
    print("Training the CBF Model on Train set...")
    # Build per-user rating histories with a single fast groupby pass instead of
    # a per-group lambda .apply(), which is slow on large datasets.
    train_user_ratings = {}
    for u, title, rating in zip(train_df['userId'], train_df['title'], train_df['rating']):
        train_user_ratings.setdefault(u, {})[title] = rating
        
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('title')['rating'].mean().to_dict()
    
    print("Evaluating Predictions (MSE, RMSE)...")
    
    # Evaluate on a stratified sample of test interactions for rapid validation
    eval_sample = test_df.sample(n=min(5000, len(test_df)), random_state=random_state)
    actuals = eval_sample['rating'].to_numpy()
    
    # Pre-resolve each user's train-history TF-IDF sub-matrix ONCE per user
    # (not once per test row) so repeat users don't redo the same lookup/slicing.
    user_hist_cache = {}  # userId -> (hist_matrix, hist_ratings_array, hist_title_set)
    
    def get_user_hist(u):
        if u in user_hist_cache:
            return user_hist_cache[u]
        user_history = train_user_ratings.get(u, {})
        hist_titles = [t for t in user_history if t in cbf_model.title_to_idx]
        if hist_titles:
            hist_idxs = [cbf_model.title_to_idx[t] for t in hist_titles]
            hist_matrix = cbf_model.tfidf_matrix[hist_idxs]           # (n_hist, n_features) sparse
            hist_ratings = np.array([user_history[t] for t in hist_titles], dtype=float)
        else:
            hist_matrix, hist_ratings, hist_titles = None, None, []
        result = (hist_matrix, hist_ratings, hist_titles)
        user_hist_cache[u] = result
        return result
    
    cb_preds = np.empty(len(eval_sample), dtype=float)
    
    for i, (u, target_movie) in enumerate(zip(eval_sample['userId'], eval_sample['title'])):
        hist_matrix, hist_ratings, hist_titles = get_user_hist(u)
        predicted = None
        
        if target_movie in cbf_model.title_to_idx and hist_matrix is not None:
            target_idx = cbf_model.title_to_idx[target_movie]
            target_vec = cbf_model.tfidf_matrix[target_idx]
            
            # Exclude the target movie itself from its own history, if present
            if target_movie in hist_titles:
                keep_mask = np.array([t != target_movie for t in hist_titles])
                calc_matrix = hist_matrix[keep_mask]
                calc_ratings = hist_ratings[keep_mask]
            else:
                calc_matrix = hist_matrix
                calc_ratings = hist_ratings
                
            if calc_matrix is not None and calc_matrix.shape[0] > 0:
                # ONE batched similarity call against the user's whole history,
                # instead of one linear_kernel call per history item.
                sims = linear_kernel(target_vec, calc_matrix).flatten()
                mask = sims > 0.05
                if mask.any():
                    predicted = float(np.dot(sims[mask], calc_ratings[mask]) / sims[mask].sum())
                    
        if predicted is None:
            predicted = movie_means.get(target_movie, global_mean)
            
        cb_preds[i] = float(np.clip(predicted, 0.5, 5.0))
    
    # Calculate Prediction Errors (MSE, RMSE)
    mse_cb = mean_squared_error(actuals, cb_preds)
    rmse_cb = sqrt(mse_cb)
    
    print(f"Evaluating Ranking Metrics (Precision@{k}, Recall@{k}, F1@{k})...")
    actual_binary = (actuals >= relevance_threshold).astype(int)
    pred_binary = (cb_preds >= relevance_threshold).astype(int)
    
    tp = ((pred_binary == 1) & (actual_binary == 1)).sum()
    fp = ((pred_binary == 1) & (actual_binary == 0)).sum()
    fn = ((pred_binary == 0) & (actual_binary == 1)).sum()
    tn = ((pred_binary == 0) & (actual_binary == 0)).sum()
    
    prec_class = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec_class = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_class = 2 * (prec_class * rec_class) / (prec_class + rec_class) if (prec_class + rec_class) > 0 else 0
    
    # Display formatted Evaluation Results
    print("\n--- Evaluation Results ---")
    print(f"MSE:          {mse_cb:.4f}")
    print(f"RMSE:         {rmse_cb:.4f}")
    print(f"Precision@{k}: {prec_class:.4f}")
    print(f"Recall@{k}:    {rec_class:.4f}")
    print(f"F1@{k}:        {f1_class:.4f}")
    print("--------------------------\n")
    
    # Render neat Boxed Table with column dividers
    results_table = pd.DataFrame([
        {"Metric": "MSE (Mean Squared Error)", "Score": f"{mse_cb:.4f}"},
        {"Metric": "RMSE (Root Mean Squared Error)", "Score": f"{rmse_cb:.4f}"},
        {"Metric": f"Precision@{k} (Relevant Recommendations)", "Score": f"{prec_class:.4f} ({prec_class*100:.2f}%)"},
        {"Metric": f"Recall@{k} (Discovered User Favorites)", "Score": f"{rec_class:.4f} ({rec_class*100:.2f}%)"},
        {"Metric": f"F1@{k} (Harmonic Mean)", "Score": f"{f1_class:.4f} ({f1_class*100:.2f}%)"}
    ])
    print_ascii_table(results_table, max_col_widths={'Metric': 42, 'Score': 32})


# ======================================================================================
# 5. DATASET & METADATA SUMMARY ANALYTICS
# ======================================================================================

def display_dataset_summary(cbf_model):
    """
    Displays metadata statistics and feature distributions.
    """
    print("\n" + "="*75)
    print("              [ANALYTICS] DATASET & CONTENT METADATA SUMMARY")
    print("="*75)
    
    stats = cbf_model.movie_stats
    num_movies = len(stats)
    num_ratings = len(cbf_model.data) if cbf_model.data is not None else 0
    vocab_size = len(cbf_model.vectorizer.vocabulary_)
    
    has_genres = (stats['genres_clean'] != '').sum()
    has_keywords = (stats['keywords_clean'] != '').sum()
    has_overview = (stats['overview_clean'] != '').sum()
    
    print(f"Total Unique Movies    : {num_movies:,}")
    print(f"Total Rating Records   : {num_ratings:,}")
    print(f"TF-IDF Vocabulary Size : {vocab_size:,} n-gram tokens")
    print(f"Metadata Completeness  :")
    print(f"  - Genres Available   : {has_genres:,} / {num_movies:,} ({has_genres/num_movies*100:.1f}%)")
    print(f"  - Keywords Available : {has_keywords:,} / {num_movies:,} ({has_keywords/num_movies*100:.1f}%)")
    print(f"  - Overviews Available: {has_overview:,} / {num_movies:,} ({has_overview/num_movies*100:.1f}%)")
    
    print("\nTop 5 Most Popular Movies by Rating Count:")
    top_popular = stats.sort_values('num_of_ratings', ascending=False).head(5)[['title', 'avg_rating', 'num_of_ratings', 'genres_clean']].copy()
    top_popular.rename(columns={'title': 'Title', 'avg_rating': 'Avg Rating', 'num_of_ratings': 'Ratings', 'genres_clean': 'Genres'}, inplace=True)
    top_popular['Avg Rating'] = top_popular['Avg Rating'].apply(lambda r: f"{r:.2f}/5.0")
    print_ascii_table(top_popular, max_col_widths={'Title': 36, 'Genres': 28})
    print("="*75 + "\n")


# ======================================================================================
# 6. INTERACTIVE CONSOLE WORKFLOWS
# ======================================================================================

def search_for_movie_mode(cbf_model):
    """
    [1] Search for a movie in the database by title, genre, keyword, or plot overview.
    """
    titles_list = cbf_model.movie_stats['title'].tolist()
    
    while True:
        print("\n" + "="*65)
        print("                 [1] SEARCH FOR A MOVIE")
        print("="*65)
        try:
            query = input("Enter movie title / keyword to search (or '0' to return): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not query:
            continue
        if query in ('0', 'b', 'back', 'exit', 'q', 'quit'):
            break
            
        matches = search_movies(query, titles_list, cbf_model.movie_stats, max_results=10)
        
        if not matches:
            print(f"\n[!] No movies found matching '{query}'.")
            print("    Tip: Try searching by keyword, genre, or partial title (e.g. 'Matrix', 'Toy Story', 'Space', 'Drama').")
            continue
            
        print(f"\nFound {len(matches)} movie(s) matching '{query}':\n")
        results = []
        for idx, title in enumerate(matches, 1):
            row = cbf_model.movie_stats[cbf_model.movie_stats['title'] == title].iloc[0]
            results.append({
                '#': idx,
                'Title': title,
                'Avg Rating': f"{row['avg_rating']:.2f}/5.0" if row['avg_rating'] > 0 else 'Unrated',
                'Ratings': row['num_of_ratings'],
                'Genres': row['genres_clean'] if row['genres_clean'] else 'N/A',
                'Overview / Plot': row['overview_clean'][:48] + '...' if len(str(row['overview_clean'])) > 48 else (row['overview_clean'] or 'N/A')
            })
        results_df = pd.DataFrame(results)
        print_ascii_table(results_df, max_col_widths={'Title': 32, 'Genres': 24, 'Overview / Plot': 42})


def get_recommendations_by_movie_mode(cbf_model):
    """
    [2] Get Content-Based recommendations for a specific movie using TF-IDF & Cosine Similarity.
    """
    titles_list = cbf_model.movie_stats['title'].tolist()
    
    while True:
        print("\n" + "="*65)
        print("          [2] GET RECOMMENDATIONS BY MOVIE")
        print("="*65)
        try:
            user_input = input("Enter movie title to get recommendations (or '0' to return): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input:
            continue
        if user_input in ('0', 'b', 'back', 'exit', 'q', 'quit'):
            break
            
        matches = search_movies(user_input, titles_list, cbf_model.movie_stats, max_results=5)
        
        if not matches:
            print(f"\n[!] No movies found matching '{user_input}'.")
            print("    Hint: Try typing a partial name (e.g. 'Matrix', 'Avatar', 'Dark Knight', 'Toy Story').")
            continue
            
        if len(matches) == 1:
            target_movie = matches[0]
        else:
            print(f"\nMultiple movies matched '{user_input}':")
            match_rows = []
            for idx, title in enumerate(matches, 1):
                row = cbf_model.movie_stats[cbf_model.movie_stats['title'] == title].iloc[0]
                match_rows.append({
                    '#': idx,
                    'Title': title,
                    'Avg Rating': f"{row['avg_rating']:.2f}/5.0" if row['avg_rating'] > 0 else 'Unrated',
                    'Genres': row['genres_clean'] or 'N/A'
                })
            print_ascii_table(pd.DataFrame(match_rows), max_col_widths={'Title': 38, 'Genres': 28})
                
            try:
                choice = input(f"\nSelect a movie [1-{len(matches)}] (default 1, '0' to cancel): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
                
            if choice in ('0', 'b', 'back'):
                continue
            if choice.lower() in ('exit', 'q', 'quit'):
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(matches):
                target_movie = matches[int(choice) - 1]
            else:
                target_movie = matches[0]
                
        while True:
            try:
                top_n_input = input("Enter number of recommendations to display [1-10, default 10]: ").strip()
            except (EOFError, KeyboardInterrupt):
                top_n = 10
                break
                
            if not top_n_input:
                top_n = 10
                break
                
            if not top_n_input.isdigit() or int(top_n_input) < 1:
                print("[!] Error: Please enter a valid positive number between 1 and 10.")
                continue
                
            val = int(top_n_input)
            if val > 10:
                print("[!] Error: The maximum recommend is 10. Please enter a number between 1 and 10.")
                continue
                
            top_n = val
            break
            
        print(f"\n[*] Computing Content Similarity (TF-IDF & Cosine) for: '{target_movie}'...\n")
        recs = cbf_model.get_similar_movies(target_movie, top_n=top_n)
        
        if recs is None or recs.empty:
            print("[!] No sufficiently similar movies found in database.")
        else:
            print(f">>> Top {len(recs)} Content-Based Recommendations for '{target_movie}':\n")
            recs_display = recs.copy()
            recs_display.insert(0, '#', range(1, len(recs_display) + 1))
            recs_display['Similarity'] = recs_display['Similarity'].apply(lambda s: f"{s:.4f}")
            recs_display['Avg Rating'] = recs_display['avg_rating'].apply(lambda r: f"{r:.2f}/5.0" if r > 0 else 'Unrated')
            recs_display.rename(columns={'genres': 'Genres', 'keywords': 'Keywords', 'title': 'Title'}, inplace=True)
            display_cols = ['#', 'Title', 'Similarity', 'Avg Rating', 'Genres', 'Keywords']
            print_ascii_table(recs_display[display_cols], max_col_widths={'Title': 34, 'Genres': 26, 'Keywords': 34})
            
            # Explain top 1 recommendation
            top_rec_title = recs.iloc[0]['title']
            explanations = cbf_model.explain_similarity(target_movie, top_rec_title)
            if explanations:
                tokens_str = ', '.join([f"'{token}' ({score})" for token, score in explanations[:4]])
                print(f"\n[i] Why recommended '{top_rec_title}'? Shared Content Tokens: {tokens_str}")


# ======================================================================================
# 7. MAIN CONTROLLER & MENU
# ======================================================================================

def main():
    print("="*65)
    print("   CONTENT-BASED FILTERING (CBF) MOVIE RECOMMENDER SYSTEM")
    print("           TF-IDF Vectorization & Cosine Similarity")
    print("="*65 + "\n")
    
    # 1. Load Data
    data = load_dataset()
    if data is None:
        return
        
    # 2. Fit Content-Based Recommender Engine
    cbf_model = ContentBasedRecommender()
    cbf_model.fit(data)
    
    # 3. Main Console Menu Loop (0 for Exit)
    while True:
        print("\n" + "-"*50)
        print("CONTENT-BASED FILTERING MENU".center(50))
        print("-"*50)
        print("[1] Search for a movie")
        print("[2] Get recommendations by movie")
        print("[3] Run Recommender System Evaluation (MSE/RMSE/Precision@K/Recall@K/F1@K)")
        print("[0] Exit")
        print("-"*50)
        
        try:
            choice = input("Enter your choice (0-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Content-Based Recommender. Goodbye!")
            break
            
        if choice == '1':
            search_for_movie_mode(cbf_model)
        elif choice == '2':
            get_recommendations_by_movie_mode(cbf_model)
        elif choice == '3':
            evaluate_content_based_system(cbf_model, data)
        elif choice in ('0', 'exit', 'quit', 'q'):
            print("\nThank you for using the Content-Based Filtering Movie Recommender. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter a number from 0 to 3.")


if __name__ == "__main__":
    main()
