import pandas as pd
from module_content_nlp import ContentNLPRecommender
from module_svd import SVDRecommender

class HybridRecommender:
    def __init__(self, movies_path='merged_movies.csv', ratings_path='ratings.csv'):
        # Initialize dependencies (Team 1 and Team 2 models)
        print("Loading Team Member 1's NLP module...")
        self.nlp_module = ContentNLPRecommender(movies_path)
        
        print("Loading Team Member 2's SVD module...")
        self.svd_module = SVDRecommender(ratings_path)
        
        self.df_movies = pd.read_csv(movies_path)

    def recommend(self, user_id, seed_movie_id, top_n=10, alpha=0.5):
        """
        Main Hybrid Recommendation Algorithm
        :param user_id: Target User ID
        :param seed_movie_id: ID of the movie the user is currently watching/likes
        :param alpha: Weight for SVD (0 to 1). 1-alpha is the weight for Content-NLP
        """
        # 1. Fetch similarity scores from Team 1's NLP module
        nlp_scores = self.nlp_module.get_similarity_scores(seed_movie_id)
        if nlp_scores is None:
            print("Error: Seed movie does not exist!")
            return pd.DataFrame()

        results = []
        
        # 2. Iterate through all movies to compute the hybrid score
        for idx, row in self.df_movies.iterrows():
            m_id = row['movieId']
            if m_id == seed_movie_id:
                continue # Skip the seed movie itself
            
            # Content Similarity Score (0 to 1)
            content_score = nlp_scores[idx]
            
            # SVD Predicted Rating, normalized to a 0-1 scale [(rating-0.5)/4.5]
            raw_svd = self.svd_module.predict_rating(user_id, m_id)
            norm_svd_score = (raw_svd - 0.5) / 4.5
            
            # Weighted Hybrid Formula
            final_hybrid_score = alpha * norm_svd_score + (1 - alpha) * content_score
            
            results.append({
                'movieId': m_id,
                'title': row['title_x'],
                'genres': row['genres'],
                'director': row['director'],
                'hybrid_score': round(final_hybrid_score, 4),
                'svd_pred_rating': round(raw_svd, 2),
                'content_sim': round(content_score, 4)
            })

        # 3. Sort by final hybrid score in descending order
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values(by='hybrid_score', ascending=False)
        return res_df.head(top_n)

# Standalone Test for Team Member 3
if __name__ == '__main__':
    hybrid_system = HybridRecommender()
    print("\n================ Hybrid Recommendation Results ================")
    # Assume recommending to userId=1, currently watching movieId=1 (Toy Story)
    recommendations = hybrid_system.recommend(user_id=1, seed_movie_id=1, top_n=5, alpha=0.6)
    print(recommendations.to_string(index=False))