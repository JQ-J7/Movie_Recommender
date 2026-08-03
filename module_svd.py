import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class SVDRecommender:
    def __init__(self, ratings_path='ratings.csv'):
        self.ratings = pd.read_csv(ratings_path)
        
        # 1. Create User-Item matrix (Rows: Users, Columns: Movies)
        self.user_item_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        
        # 2. Calculate average rating for each user
        self.user_means = self.ratings.groupby('userId')['rating'].mean().to_dict()
        
        # 3. Apply TruncatedSVD from scikit-learn for Matrix Factorization
        n_components = min(20, self.user_item_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_
        
        # 4. Reconstruct the predicted rating matrix
        self.predicted_matrix = np.dot(self.user_factors, self.item_factors)
        
        # Build index for fast querying
        self.pred_df = pd.DataFrame(
            self.predicted_matrix, 
            index=self.user_item_matrix.index, 
            columns=self.user_item_matrix.columns
        )

    def predict_rating(self, user_id, movie_id):
        """Interface for Team Member 3: Predict the rating user_id would give to movie_id"""
        # If user or movie is unknown, return a baseline average
        if user_id not in self.pred_df.index:
            return 3.0
        if movie_id not in self.pred_df.columns:
            return float(self.user_means.get(user_id, 3.0))
        
        # Extract predicted score
        pred_score = self.pred_df.loc[user_id, movie_id]
        
        # If prediction is unreasonably low, fallback to user's historical average
        if pred_score <= 0.5:
            pred_score = self.user_means.get(user_id, 3.0)
            
        # Clip the final score to be within the standard 0.5 to 5.0 range
        return float(np.clip(pred_score, 0.5, 5.0))

    def recommend_standalone(self, user_id, all_movie_ids, top_n=5):
        """Standalone test method for Team Member 2"""
        predictions = []
        for m_id in all_movie_ids:
            pred_score = self.predict_rating(user_id, m_id)
            predictions.append((m_id, pred_score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

# Standalone Test for Team Member 2
if __name__ == '__main__':
    svd_rec = SVDRecommender()
    print("--- Team Member 2 Test: Predict rating of userId=1 for movieId=1 ---")
    score = svd_rec.predict_rating(user_id=1, movie_id=1)
    print(f"Predicted rating: {score:.2f} / 5.0")