import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentNLPRecommender:
    def __init__(self, movies_path='merged_movies.csv'):
        self.df = pd.read_csv(movies_path).fillna('')
        
        # 1. Create a "soup" of text features (Genres + Director + Top Cast)
        self.df['soup'] = self.df['genres'] + ' ' + self.df['director'] + ' ' + self.df['top_cast']
        
        # 2. Text Vectorization using TF-IDF
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['soup'])
        
        # 3. Compute Cosine Similarity Matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Map movieId to DataFrame index for easy lookup
        self.id_to_idx = pd.Series(self.df.index, index=self.df['movieId']).to_dict()

    def get_similarity_scores(self, target_movie_id):
        """Interface for Team Member 3: Returns the similarity score vector for a specific movie"""
        if target_movie_id not in self.id_to_idx:
            return None
        idx = self.id_to_idx[target_movie_id]
        return self.cosine_sim[idx]

    def recommend_standalone(self, target_movie_id, top_n=5):
        """Standalone test method for Team Member 1"""
        scores = self.get_similarity_scores(target_movie_id)
        if scores is None:
            return "Movie not found"
        
        # Sort and extract top N similar movies
        sim_indices = list(enumerate(scores))
        sim_indices = sorted(sim_indices, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        result_indices = [i[0] for i in sim_indices]
        return self.df.iloc[result_indices][['movieId', 'title_x', 'genres', 'director']]

# Standalone Test for Team Member 1
if __name__ == '__main__':
    nlp_rec = ContentNLPRecommender()
    print("--- Team Member 1 Test: Movies similar to movieId=1 (Toy Story) ---")
    print(nlp_rec.recommend_standalone(target_movie_id=1, top_n=5))