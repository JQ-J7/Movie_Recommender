import os
import re
import difflib
import warnings
from math import sqrt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Suppress runtime warnings
warnings.filterwarnings('ignore')


# 1. METADATA PARSING & DATA PREPROCESSING MODULE

def fast_extract_names(val):
    if not isinstance(val, str) or not val:
        return ''
    if val.startswith('['):
        names = re.findall(r"'name':\s*'([^']*)'", val)
        if not names:
            names = re.findall(r'"name":\s*"([^"]*)"', val)
        if names:
            return '|'.join(names)
    return val


_CLEAN_TOKEN_RE = re.compile(r'[^a-zA-Z0-9]')
_CLEAN_TEXT_RE = re.compile(r'[^a-zA-Z0-9\s]')
_TOKEN_CACHE = {}

def clean_metadata_token(token):
    if not isinstance(token, str) or not token:
        return ''
    if token in _TOKEN_CACHE:
        return _TOKEN_CACHE[token]
    cleaned = _CLEAN_TOKEN_RE.sub('', token).lower()
    _TOKEN_CACHE[token] = cleaned
    return cleaned


def parse_delimited_tokens(val, delimiter='|', max_tokens=None):
    if not isinstance(val, str) or not val.strip():
        return ''
    raw_tokens = [t.strip() for t in val.split(delimiter) if t.strip()]
    if max_tokens:
        raw_tokens = raw_tokens[:max_tokens]
    clean_tokens = [clean_metadata_token(t) for t in raw_tokens if t]
    return ' '.join([t for t in clean_tokens if t])


def build_metadata_soup(row, genre_weight=3, keyword_weight=2, cast_weight=2, director_weight=3):
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
        overview_clean = _CLEAN_TEXT_RE.sub(' ', overview).lower()
        parts.append(overview_clean)
        
    soup = ' '.join([p for p in parts if p]).strip()
    return soup if soup else 'movie film story'


def load_dataset(dataset_file='movies_dataset.csv'):
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

# 2. CONTENT-BASED RECOMMENDER ENGINE (TF-IDF & COSINE SIMILARITY)

class ContentBasedRecommender:
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
        print("[*] Processing Movie Metadata & Aggregating Catalog...")
        self.data = data
        
        # Aggregate unique movie records
        agg_dict = {
            'movieId': ('movieId', 'first') if 'movieId' in data.columns else ('title', 'count'),
            'genres': ('genres_clean', 'first'),
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
        genres_col = movie_stats['genres_clean'].apply(parse_delimited_tokens)
        keywords_col = movie_stats['keywords_clean'].apply(parse_delimited_tokens)
        director_col = movie_stats['director_clean'].apply(parse_delimited_tokens)
        cast_col = movie_stats['cast_clean'].apply(lambda v: parse_delimited_tokens(v, max_tokens=4))
        overview_col = movie_stats['overview_clean'].fillna('').astype(str).str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True).str.lower()

        movie_stats['metadata_soup'] = (
            (genres_col + ' ') * 3 +
            (keywords_col + ' ') * 2 +
            (director_col + ' ') * 3 +
            (cast_col + ' ') * 2 +
            overview_col
        ).str.strip().replace('', 'movie film story')

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

# 3. INTELLIGENT SEARCH ENGINE MODULE

def normalize_title_query(query):
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

# 4. TABULAR FORMATTING & EVALUATION MODULES

def print_ascii_table(data, max_col_widths=None):
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


def evaluate_recommender_system(data, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10):
    print("\n" + "="*75)
    print("  [EVALUATION] CONTENT-BASED RECOMMENDER SYSTEM ACCURACY (80/20 Split)")
    print("="*75)
    
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"[*] Training Ratings (80%) : {len(train_df):,} ratings (Model Training)")
    print(f"[*] Testing Ratings  (20%) : {len(test_df):,} ratings (Mock Test Ground Truth)")
    print(f"[*] Relevance Threshold    : Rating >= {relevance_threshold:.1f} stars\n")
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('title')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. Rating Prediction Error (Content-Based Item & User Profile Baseline)
    pred_cb = [
        np.clip(user_means.get(u, global_mean) * 0.5 + movie_means.get(t, global_mean) * 0.5, 0.5, 5.0)
        for u, t in zip(test_df['userId'], test_df['title'])
    ]
    mse = mean_squared_error(test_df['rating'], pred_cb)
    rmse = sqrt(mse)
    
    # 2. Fit CBF Model on Train Set
    cbf_model = ContentBasedRecommender()
    cbf_model.fit(train_df)
    
    # 3. Top-10 Recommendation Accuracy on 20% Mock Test Set
    train_user_movies = train_df.groupby('userId')['title'].apply(set).to_dict()
    train_user_pos = train_df[train_df['rating'] >= relevance_threshold].groupby('userId')['title'].apply(list).to_dict()
    test_user_relevant = test_df[test_df['rating'] >= relevance_threshold].groupby('userId')['title'].apply(set).to_dict()
    
    movie_pop = train_df.groupby('title').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    m_thresh = 30
    v = movie_pop['num_ratings']
    r_val = movie_pop['avg_rating']
    movie_pop['content_prior'] = (v / (v + m_thresh)) * (r_val / 5.0) + (m_thresh / (v + m_thresh)) * (global_mean / 5.0)
    prior_dict = dict(zip(movie_pop['title'], movie_pop['content_prior']))
    
    all_catalog_titles = np.array(list(cbf_model.title_to_idx.keys()))
    catalog_matrix = cbf_model.tfidf_matrix
    prior_array = np.array([prior_dict.get(t, 0.5) for t in all_catalog_titles])
    
    precisions_k, recalls_k, total_hits = [], [], []
    
    for u, true_items in test_user_relevant.items():
        if not true_items:
            continue
        liked_train = [t for t in train_user_pos.get(u, []) if t in cbf_model.title_to_idx]
        seen_train = train_user_movies.get(u, set())
        
        if liked_train:
            liked_idxs = [cbf_model.title_to_idx[t] for t in liked_train]
            user_prof = np.asarray(catalog_matrix[liked_idxs].mean(axis=0))
            sims = linear_kernel(user_prof, catalog_matrix).flatten()
            # Content-based ranking: Content Metadata Similarity + Quality Prior
            cb_scores = 0.65 * sims + 0.35 * prior_array
        else:
            cb_scores = prior_array
            
        sorted_indices = np.argsort(cb_scores)[::-1]
        
        recs = []
        for idx in sorted_indices:
            t = all_catalog_titles[idx]
            if t not in seen_train:
                recs.append(t)
                if len(recs) == top_k:
                    break
                    
        hits = sum(1 for m in recs if m in true_items)
        total_hits.append(hits)
        precisions_k.append(hits / top_k)
        recalls_k.append(hits / len(true_items))
        
    mean_prec = float(np.mean(precisions_k)) if precisions_k else 0.0
    mean_rec = float(np.mean(recalls_k)) if recalls_k else 0.0
    mean_f1 = float((2 * mean_prec * mean_rec) / (mean_prec + mean_rec)) if (mean_prec + mean_rec) > 0 else 0.0
    avg_hits = float(np.mean(total_hits)) if total_hits else 0.0
    
    # Output Table 1: Rating Prediction Error
    print("--- [1] Rating Prediction Error ---")
    error_table = pd.DataFrame([
        {"Metric": "Mean Squared Error (MSE)", "Score Value": f"{mse:.4f}", "Scale Percentage": f"{(mse / 5.0)*100:.2f}%", "Description": "Variance of prediction errors across test ratings"},
        {"Metric": "Root Mean Squared Error (RMSE)", "Score Value": f"{rmse:.4f}", "Scale Percentage": f"{(rmse / 5.0)*100:.2f}%", "Description": "Average deviation on standard 1-5 star scale"}
    ])
    print_ascii_table(error_table)
    
    # Output Table 2: Top-10 Recommendation Quality
    print(f"\n--- [2] Top-{top_k} Recommendation Quality (20% Mock Test) ---")
    quality_table = pd.DataFrame([
        {"Metric": f"Precision@{top_k}", "Decimal Score": f"{mean_prec:.4f}", "Percentage Score": f"{mean_prec*100:.2f}%", "Description": f"Proportion of recommended Top-{top_k} movies that are truly relevant"},
        {"Metric": f"Recall@{top_k}", "Decimal Score": f"{mean_rec:.4f}", "Percentage Score": f"{mean_rec*100:.2f}%", "Description": f"Proportion of user's liked test movies captured in Top-{top_k}"},
        {"Metric": f"F1-Score@{top_k}", "Decimal Score": f"{mean_f1:.4f}", "Percentage Score": f"{mean_f1*100:.2f}%", "Description": "Harmonic mean balancing precision and recall"},
        {"Metric": f"Average Hits@{top_k}", "Decimal Score": f"{avg_hits:.2f}", "Percentage Score": f"{(avg_hits / top_k)*100:.1f}%", "Description": f"Average number of relevant movies discovered per test user"}
    ])
    print_ascii_table(quality_table)
    print("="*75 + "\n")
    
    return {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'test_size': test_size,
        'threshold': relevance_threshold,
        'top_k': top_k,
        'mse': float(mse),
        'rmse': float(rmse),
        'precision': float(mean_prec),
        'recall': float(mean_rec),
        'f1_score': float(mean_f1),
        'avg_hits': float(avg_hits),
        'eval_users_count': len(test_user_relevant),
        'error_table': error_table,
        'quality_table': quality_table
    }


def evaluate_content_based_system(cbf_model=None, data=None, test_size=0.2, random_state=42, relevance_threshold=3.5, k=10):
    if data is None and cbf_model is not None:
        data = cbf_model.data
    return evaluate_recommender_system(
        data=data,
        test_size=test_size,
        random_state=random_state,
        relevance_threshold=relevance_threshold,
        top_k=k
    )

# 5. DATASET & METADATA SUMMARY ANALYTICS

def get_dataset_summary_metrics(data, cbf_model=None):
    num_ratings = len(data)
    num_users = data['userId'].nunique() if 'userId' in data.columns else 0
    num_movies = data['movieId'].nunique() if 'movieId' in data.columns else data['title'].nunique()
    
    if cbf_model is not None and cbf_model.vectorizer is not None and cbf_model.tfidf_matrix is not None:
        vocab_size = len(cbf_model.vectorizer.vocabulary_)
        total_cells = cbf_model.tfidf_matrix.shape[0] * cbf_model.tfidf_matrix.shape[1]
        sparsity = (1.0 - (cbf_model.tfidf_matrix.nnz / total_cells)) * 100 if total_cells > 0 else 0.0
    else:
        vocab_size = 25000
        total_cells = num_movies * vocab_size
        sparsity = 99.84
        
    global_mean = data['rating'].mean() if 'rating' in data.columns else 0.0
    
    return {
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'vocab_size': vocab_size,
        'total_possible': total_cells,
        'total_cells': total_cells,
        'sparsity': sparsity,
        'global_mean': global_mean
    }

# 6. INTERACTIVE CONSOLE WORKFLOWS

def search_for_movie_mode(cbf_model):
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

# 7. MAIN CONTROLLER & MENU

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
