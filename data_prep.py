import pandas as pd
import json
import os

def prepare_data():
    print("Merging and cleaning data, please wait...")
    
    # 1. Read base MovieLens files
    df_movies = pd.read_csv('movies.csv')
    df_links = pd.read_csv('links.csv')

    df_links_clean = df_links.dropna(subset=['tmdbId']).copy()
    df_links_clean['tmdbId'] = df_links_clean['tmdbId'].astype(int)

    df_merged = df_movies.merge(df_links_clean, on='movieId')

    # 2. Automatically check and read either tmdb_5000_movies.csv or tmdb_5000_credits.csv
    if os.path.exists('tmdb_5000_movies.csv'):
        df_tmdb = pd.read_csv('tmdb_5000_movies.csv')
        df_final = df_merged.merge(df_tmdb, left_on='tmdbId', right_on='id')
    elif os.path.exists('tmdb_5000_credits.csv'):
        df_tmdb = pd.read_csv('tmdb_5000_credits.csv')
        df_final = df_merged.merge(df_tmdb, left_on='tmdbId', right_on='movie_id')
    else:
        df_final = df_merged

    # 3. Parse director and cast (fill with empty string if missing)
    def extract_director(crew_str):
        if not isinstance(crew_str, str) or not crew_str:
            return ''
        try:
            crew_list = json.loads(crew_str)
            for person in crew_list:
                if person.get('job') == 'Director':
                    return person['name']
        except Exception:
            return ''
        return ''

    def extract_top_cast(cast_str, top_n=3):
        if not isinstance(cast_str, str) or not cast_str:
            return ''
        try:
            cast_list = json.loads(cast_str)
            return ' '.join([person['name'].replace(' ', '') for person in cast_list[:top_n]])
        except Exception:
            return ''

    if 'crew' in df_final.columns:
        df_final['director'] = df_final['crew'].apply(extract_director).str.replace(' ', '')
    else:
        df_final['director'] = ''

    if 'cast' in df_final.columns:
        df_final['top_cast'] = df_final['cast'].apply(extract_top_cast)
    else:
        df_final['top_cast'] = ''

    # Clean genres and titles
    if 'genres_x' in df_final.columns:
        df_final['genres'] = df_final['genres_x'].fillna('').str.replace('|', ' ', regex=False)
    elif 'genres' in df_final.columns:
        df_final['genres'] = df_final['genres'].fillna('').str.replace('|', ' ', regex=False)
    else:
        df_final['genres'] = ''

    if 'title_x' in df_final.columns:
        df_final['title_x'] = df_final['title_x']
    elif 'title' in df_final.columns:
        df_final['title_x'] = df_final['title']

    output_cols = ['movieId', 'tmdbId', 'title_x', 'genres', 'director', 'top_cast']
    final_cols = [col for col in output_cols if col in df_final.columns]
    
    df_out = df_final[final_cols].drop_duplicates(subset=['movieId'])
    df_out.to_csv('merged_movies.csv', index=False)
    print("✅ Successfully generated merged_movies.csv!")

if __name__ == '__main__':
    prepare_data()