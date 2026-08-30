import os
import re
import difflib
import warnings
from math import sqrt, log2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ignore runtime warnings
warnings.filterwarnings('ignore')


# extract names from json-like strings or list of dicts
def fast_extract_names(val):
    if not isinstance(val, str) or not val:
        return ''
    if val.startswith('['):
        names = re.findall(r"'name':\s*'([^']*)'", val)
        if names:
            return '|'.join(names)
    return val


# load dataset and clean metadata columns
def load_dataset(dataset_file='movies_dataset.csv'):
    try:
        if not os.path.exists(dataset_file):
            print(f"Error: Dataset file '{dataset_file}' not found.")
            print(f"Please make sure '{dataset_file}' is in the current directory.")
            return None
            
        print(f"Loading dataset from '{dataset_file}'...")
        data = pd.read_csv(dataset_file)
        
        # clean text fields
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
            
        print(f"Loaded {len(data):,} ratings across {data['movieId'].nunique():,} unique movies.\n")
        return data
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# build user-item matrix and calculate movie stats
def build_recommender_matrix(data):
    print("Building user-item matrix and movie statistics...")
    
    # calculate movie stats
    movie_stats = data.groupby('title').agg(
        avg_rating=('rating', 'mean'),
        num_of_ratings=('rating', 'count'),
        genres=('genres_clean', 'first'),
        keywords=('keywords_clean', 'first'),
        overview=('overview_clean', 'first'),
        movieId=('movieId', 'first')
    ).reset_index()
    
    # pivot table: users as rows, movies as columns
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating', aggfunc='mean')
    
    num_users, num_movies = user_movie_matrix.shape
    print(f"User-Item Matrix shape: {num_users} users x {num_movies} movies.\n")
    return user_movie_matrix, movie_stats


# handle common movie prefixes like 'The', 'A', 'An'
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


# search movies by title, keywords, genres, and overview
def search_movies(query, titles_list, movie_stats, max_results=5):
    query_clean = query.strip()
    query_lower = query_clean.lower()
    query_variants = normalize_title_query(query_clean)
    stats_map = dict(zip(movie_stats['title'], movie_stats['num_of_ratings']))
    
    # 1. exact match on title variants
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if title.lower() == var_lower:
                return [title]
            clean_title = re.sub(r'\s*\(\d{4}\)', '', title).strip().lower()
            if clean_title == var_lower:
                return [title]
                
    # 2. search across title, keywords, genres, and overview
    scored_candidates = {}
    
    # title substring match
    for var in query_variants:
        var_lower = var.lower()
        for title in titles_list:
            if var_lower in title.lower():
                pop = stats_map.get(title, 0)
                scored_candidates[title] = max(scored_candidates.get(title, 0), 1000 + pop)
                
    # keyword match
    if 'keywords' in movie_stats.columns:
        kw_matches = movie_stats[movie_stats['keywords'].str.contains(query_clean, case=False, na=False, regex=False)]
        for _, row in kw_matches.iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 500 + pop)
            
    # genre match
    genre_matches = movie_stats[movie_stats['genres'].str.contains(query_clean, case=False, na=False, regex=False)]
    for _, row in genre_matches.iterrows():
        t = row['title']
        pop = row['num_of_ratings']
        scored_candidates[t] = max(scored_candidates.get(t, 0), 200 + pop)
        
    # overview match
    if 'overview' in movie_stats.columns:
        ov_matches = movie_stats[movie_stats['overview'].str.contains(r'\b' + re.escape(query_clean) + r'\b', case=False, na=False, regex=True)]
        for _, row in ov_matches.head(10).iterrows():
            t = row['title']
            pop = row['num_of_ratings']
            scored_candidates[t] = max(scored_candidates.get(t, 0), 100 + pop)
            
    if scored_candidates:
        sorted_candidates = sorted(scored_candidates.keys(), key=lambda t: scored_candidates[t], reverse=True)
        return sorted_candidates[:max_results]
        
    # 3. fuzzy search fallback
    fuzzy_matches = difflib.get_close_matches(query_clean, titles_list, n=max_results, cutoff=0.4)
    return fuzzy_matches


# item-based collaborative filtering recommender class
class CollaborativeRecommender:
    def __init__(self, min_ratings=50, min_overlap=15):
        self.min_ratings = min_ratings
        self.min_overlap = min_overlap
        self.data = None
        self.user_movie_matrix = None
        self.movie_stats = None
        self.titles_list = []

    # fit model on dataset
    def fit(self, data):
        self.data = data
        self.user_movie_matrix, self.movie_stats = build_recommender_matrix(data)
        if self.movie_stats is not None and 'title' in self.movie_stats.columns:
            self.titles_list = self.movie_stats['title'].tolist()
        return self

    # calculate pearson correlation scores for hybrid model
    def compute_similarity_scores(self, target_title, min_overlap=None):
        if min_overlap is None:
            min_overlap = self.min_overlap

        if self.movie_stats is None or self.user_movie_matrix is None:
            return pd.Series(dtype=float)

        all_titles = self.movie_stats['title']
        cf_scores = pd.Series(0.0, index=all_titles)

        if target_title not in self.user_movie_matrix.columns:
            return cf_scores

        target_ratings = self.user_movie_matrix[target_title]
        target_mask = target_ratings.notna()

        # filter candidates by minimum rating count for speed
        candidate_titles = self.movie_stats[self.movie_stats['num_of_ratings'] >= min(5, self.min_ratings)]['title']
        candidate_cols = [c for c in candidate_titles if c in self.user_movie_matrix.columns]
        candidate_matrix = self.user_movie_matrix[candidate_cols]

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
                        # scale pearson r from [-1, 1] to [0, 1]
                        corrs[col] = (r + 1.0) / 2.0

        for col, score in corrs.items():
            if col in cf_scores.index:
                cf_scores[col] = score

        return cf_scores

    # get top n recommendations for a movie
    def get_similar_movies(self, movie_title, top_n=10, min_ratings=None, min_overlap=None):
        m_ratings = min_ratings if min_ratings is not None else self.min_ratings
        m_overlap = min_overlap if min_overlap is not None else self.min_overlap
        return get_collaborative_recommendations(
            movie_title,
            self.user_movie_matrix,
            self.movie_stats,
            min_ratings=m_ratings,
            min_overlap=m_overlap,
            top_n=top_n
        )

    # search catalog
    def search(self, query, max_results=5):
        if self.movie_stats is None:
            return []
        return search_movies(query, self.titles_list, self.movie_stats, max_results=max_results)

    # run offline evaluation with 80/20 train-test split
    def evaluate(self, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10):
        if self.data is None:
            print("Model must be fitted with dataset before running evaluation.")
            return None
        return evaluate_recommender_system(
            self.data,
            test_size=test_size,
            random_state=random_state,
            relevance_threshold=relevance_threshold,
            top_k=top_k
        )


# calculate item-item pearson correlation recommendations
def get_collaborative_recommendations(movie_title, user_movie_matrix, movie_stats, min_ratings=50, min_overlap=15, top_n=10):
    if movie_title not in user_movie_matrix.columns:
        return None
        
    target_ratings = user_movie_matrix[movie_title]
    target_non_null = target_ratings.notna()
    
    # filter movies with at least min_ratings
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
    
    # sort by correlation score and rating count
    recommendations = results.sort_values(by=['Correlation', 'num_of_ratings'], ascending=[False, False])
    
    output_cols = ['title', 'Correlation', 'avg_rating', 'num_of_ratings', 'Co-rated Users', 'genres']
    return recommendations[output_cols].head(top_n).reset_index(drop=True)


# print table with borders
def print_ascii_table(headers, rows, alignments=None):
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


# print recommendations in formatted table
def print_recommendations_table(df):
    if df is None or df.empty:
        print("No recommendations found meeting the correlation threshold.")
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


# evaluate rating error (MSE, RMSE) and top-k metrics (Precision, Recall, F1) on 80/20 split
def evaluate_recommender_system(data, test_size=0.2, random_state=42, relevance_threshold=3.5, top_k=10):
    print("\n" + "="*70)
    print("      Evaluation: Collaborative Filtering (80/20 Train-Test Split)")
    print("="*70)
    
    # 80/20 split
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    print(f"Training set: {len(train_df):,} ratings (80%)")
    print(f"Testing set : {len(test_df):,} ratings (20%)")
    print(f"Relevance threshold: >= {relevance_threshold:.1f} stars\n")
    
    global_mean = train_df['rating'].mean()
    movie_means = train_df.groupby('movieId')['rating'].mean().to_dict()
    user_means = train_df.groupby('userId')['rating'].mean().to_dict()
    
    # 1. rating prediction error using baseline predictor
    pred_combined = [
        np.clip(user_means.get(u, global_mean) + movie_means.get(m, global_mean) - global_mean, 0.5, 5.0)
        for u, m in zip(test_df['userId'], test_df['movieId'])
    ]
    mse = mean_squared_error(test_df['rating'], pred_combined)
    rmse = sqrt(mse)
    
    # 2. top-10 recommendation metrics on test set
    train_user_movies = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    test_user_relevant = test_df[test_df['rating'] >= relevance_threshold].groupby('userId')['movieId'].apply(set).to_dict()
    
    movie_pop_stats = train_df.groupby('movieId').agg(
        num_ratings=('rating', 'count'),
        avg_rating=('rating', 'mean')
    ).reset_index()
    
    # weighted rating formula for ranking
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
        
    mean_prec = float(np.mean(precisions_k)) if precisions_k else 0.0
    mean_rec = float(np.mean(recalls_k)) if recalls_k else 0.0
    mean_f1 = float((2 * mean_prec * mean_rec) / (mean_prec + mean_rec)) if (mean_prec + mean_rec) > 0 else 0.0
    avg_hits = float(np.mean(total_hits)) if total_hits else 0.0
    
    # print prediction error table
    print("--- Rating Prediction Error ---")
    headers_1 = ["Error Metric", "Score Value", "Percentage"]
    rows_1 = [
        ["Mean Squared Error (MSE)", f"{mse:.4f}", f"{(mse / 5.0)*100:.2f}%"],
        ["Root Mean Squared Error (RMSE)", f"{rmse:.4f}", f"{(rmse / 5.0)*100:.2f}%"]
    ]
    print_ascii_table(headers_1, rows_1, alignments=['left', 'center', 'center'])
    
    # print ranking quality table
    print(f"\n--- Top-{top_k} Recommendation Quality ---")
    headers_2 = [f"Top-{top_k} Metric", "Score Value", "Percentage"]
    rows_2 = [
        [f"Precision@{top_k}", f"{mean_prec:.4f}", f"{mean_prec*100:.2f}%"],
        [f"Recall@{top_k}", f"{mean_rec:.4f}", f"{mean_rec*100:.2f}%"],
        [f"F1-Score@{top_k}", f"{mean_f1:.4f}", f"{mean_f1*100:.2f}%"]
    ]
    print_ascii_table(headers_2, rows_2, alignments=['left', 'center', 'center'])
    print("="*70 + "\n")
    
    error_table = pd.DataFrame([
        {"Metric": "Mean Squared Error (MSE)", "Score Value": f"{mse:.4f}", "Scale Percentage": f"{(mse / 5.0)*100:.2f}%", "Description": "Variance of prediction errors across test ratings"},
        {"Metric": "Root Mean Squared Error (RMSE)", "Score Value": f"{rmse:.4f}", "Scale Percentage": f"{(rmse / 5.0)*100:.2f}%", "Description": "Average deviation on standard 1-5 star scale"}
    ])
    
    quality_table = pd.DataFrame([
        {"Metric": f"Precision@{top_k}", "Decimal Score": f"{mean_prec:.4f}", "Percentage Score": f"{mean_prec*100:.2f}%", "Description": f"Proportion of recommended Top-{top_k} movies that are truly relevant"},
        {"Metric": f"Recall@{top_k}", "Decimal Score": f"{mean_rec:.4f}", "Percentage Score": f"{mean_rec*100:.2f}%", "Description": f"Proportion of user's liked test movies captured in Top-{top_k}"},
        {"Metric": f"F1-Score@{top_k}", "Decimal Score": f"{mean_f1:.4f}", "Percentage Score": f"{mean_f1*100:.2f}%", "Description": "Harmonic mean balancing precision and recall"},
        {"Metric": f"Average Hits@{top_k}", "Decimal Score": f"{avg_hits:.2f}", "Percentage Score": f"{(avg_hits / top_k)*100:.1f}%", "Description": f"Average number of relevant movies discovered per test user"}
    ])
    
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


# compute dataset statistics and matrix sparsity
def get_dataset_summary_metrics(data):
    num_ratings = len(data)
    num_users = data['userId'].nunique()
    num_movies = data['movieId'].nunique() if 'movieId' in data.columns else data['title'].nunique()
    total_possible = num_users * num_movies
    sparsity = (1.0 - (num_ratings / total_possible)) * 100 if total_possible > 0 else 0.0
    global_mean = data['rating'].mean()
    
    return {
        'num_ratings': num_ratings,
        'num_users': num_users,
        'num_movies': num_movies,
        'total_possible': total_possible,
        'sparsity': sparsity,
        'global_mean': global_mean
    }


# print dataset summary in console
def display_dataset_summary(data):
    print("\n" + "="*70)
    print("                 Dataset Summary & Statistics")
    print("="*70)
    
    stats = get_dataset_summary_metrics(data)
    
    print(f"Total Ratings      : {stats['num_ratings']:,}")
    print(f"Unique Users       : {stats['num_users']:,}")
    print(f"Unique Movies      : {stats['num_movies']:,}")
    print(f"Rating Range       : {data['rating'].min()} to {data['rating'].max()} stars")
    print(f"Average Rating     : {stats['global_mean']:.2f} stars")
    print(f"Matrix Dimensions  : {stats['num_users']} users x {stats['num_movies']} movies ({stats['total_possible']:,} cells)")
    print(f"Matrix Sparsity    : {stats['sparsity']:.2f}%")
    print("="*70 + "\n")


# interactive search for movie details
def interactive_movie_search_only(movie_stats):
    titles_list = movie_stats['title'].tolist()
    
    while True:
        print("\n" + "-"*70)
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
            print(f"No movies found matching '{user_input}'.")
            print("Hint: Try a keyword like 'Matrix', 'Star Wars', 'Avatar', 'Batman'.")
            continue
            
        # handle multiple matches
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
                
        # display details
        movie_row = movie_stats[movie_stats['title'] == target_movie].iloc[0]
        print("\n" + "="*70)
        print("                          Movie Details")
        print("="*70)
        print(f"Title          : {movie_row['title']}")
        print(f"Movie ID       : {movie_row['movieId']}")
        print(f"Average Rating : {movie_row['avg_rating']:.2f} / 5.0 stars")
        print(f"Total Ratings  : {movie_row['num_of_ratings']:,} ratings")
        print(f"Genres         : {movie_row['genres'].replace('|', ' | ') if movie_row['genres'] else 'N/A'}")
        if 'keywords' in movie_row and movie_row['keywords']:
            print(f"Keywords       : {movie_row['keywords'].replace('|', ' | ')}")
        if 'overview' in movie_row and movie_row['overview']:
            print(f"Overview       : {movie_row['overview']}")
        print("="*70)


# interactive movie recommendation search
def interactive_search_mode(user_movie_matrix, movie_stats):
    titles_list = movie_stats['title'].tolist()
    
    while True:
        print("\n" + "-"*70)
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
            print(f"No movies found matching '{user_input}'.")
            print("Hint: Try a keyword like 'Matrix', 'Avatar', 'Batman', 'Toy Story'.")
            continue
            
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
                
        try:
            num_recs_input = input("Enter number of recommendations to display [default 10]: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if num_recs_input.lower() in ('b', 'back', 'cancel', 'c'):
            continue
        if num_recs_input.lower() in ('exit', 'q', 'quit'):
            break
            
        top_n = int(num_recs_input) if num_recs_input.isdigit() and int(num_recs_input) > 0 else 10
        
        print(f"\nGenerating recommendations for: '{target_movie}' (Top {top_n})...\n")
        recs = get_collaborative_recommendations(target_movie, user_movie_matrix, movie_stats, min_ratings=50, top_n=top_n)
        
        if recs is None or recs.empty:
            print("No recommendations found meeting the correlation threshold.")
        else:
            print("Top Recommendations:")
            print_recommendations_table(recs)


# main console menu
def main():
    print("="*70)
    print("          Collaborative Filtering Movie Recommender System")
    print("="*70 + "\n")
    
    # 1. load data
    data = load_dataset()
    if data is None:
        return
        
    # 2. build matrix
    user_movie_matrix, movie_stats = build_recommender_matrix(data)
    
    # 3. main loop
    while True:
        print("\n" + "="*45)
        print("                  MAIN MENU")
        print("="*45)
        print("  [1] Search Movie (View Details & Ratings)")
        print("  [2] Get Recommendations by Movie")
        print("  [3] Run Evaluation (RMSE/MSE/Precision/Recall/F1)")
        print("  [4] View Dataset Summary & Statistics")
        print("  [5] Exit")
        print("="*45)
        
        try:
            choice = input("Enter your option [1-5]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
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
            print("Invalid option. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()

