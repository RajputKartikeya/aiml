"""
🤖 Movie Recommendation Engine

This is where the magic happens! We'll implement collaborative filtering,
which is the most common approach for recommendation systems.

How does collaborative filtering work?
1. Find users who have similar tastes to you
2. Look at what those similar users liked
3. Recommend those movies to you!

It's like asking friends with similar taste: "What movies did you like that I haven't seen?"
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CollaborativeFilteringRecommender:
    """
    A beginner-friendly collaborative filtering recommendation system.
    
    This class implements two main approaches:
    1. User-based: Find similar users and recommend what they liked
    2. Item-based: Find similar movies and recommend based on that
    """
    
    def __init__(self, min_ratings: int = 5):
        """
        Initialize our recommendation engine.
        
        Args:
            min_ratings: Minimum number of ratings needed to make recommendations
        """
        self.min_ratings = min_ratings
        self.user_movie_matrix = None
        self.movies_df = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.is_trained = False
        
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
        """
        Train our recommendation system on the data.
        
        What happens during training?
        1. Create a user-movie matrix
        2. Calculate user similarities (who likes similar movies?)
        3. Calculate movie similarities (which movies are liked by similar people?)
        
        Args:
            ratings_df: DataFrame with userId, movieId, rating columns
            movies_df: DataFrame with movieId, title, genres columns
        """
        print("🏋️ Training recommendation system...")
        
        # Store the movies data for later use
        self.movies_df = movies_df
        
        # Create user-movie matrix
        print("📊 Creating user-movie matrix...")
        self.user_movie_matrix = ratings_df.pivot_table(
            index='userId',
            columns='movieId', 
            values='rating',
            fill_value=0
        )
        
        print(f"Matrix shape: {self.user_movie_matrix.shape}")
        
        # Calculate user similarities
        print("👥 Calculating user similarities...")
        self._calculate_user_similarities()
        
        # Calculate item similarities  
        print("🎬 Calculating movie similarities...")
        self._calculate_item_similarities()
        
        self.is_trained = True
        print("✅ Training complete!")
        
    def _calculate_user_similarities(self):
        """
        Calculate how similar users are to each other.
        
        We use cosine similarity, which measures the angle between user preference vectors.
        
        Why cosine similarity?
        - It focuses on the pattern of ratings, not the absolute values
        - User A: [5,4,5,3] and User B: [4,3,4,2] have similar patterns (both high->low->high->medium)
        - Even though B rates everything lower, they have similar taste!
        """
        # Get the user-movie matrix as a numpy array
        user_matrix = self.user_movie_matrix.values
        
        # Calculate cosine similarity between all users
        # This creates a matrix where user_similarity[i][j] = how similar user i is to user j
        self.user_similarity_matrix = cosine_similarity(user_matrix)
        
        # Convert to DataFrame for easier handling
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
        print(f"👯 Calculated similarities for {len(self.user_similarity_matrix)} users")
        
    def _calculate_item_similarities(self):
        """
        Calculate how similar movies are to each other.
        
        We look at which users rated both movies similarly.
        If Movie A and Movie B are consistently rated similarly by users,
        then they're similar movies!
        """
        # Transpose the matrix so movies are rows and users are columns
        movie_matrix = self.user_movie_matrix.T.values
        
        # Calculate cosine similarity between all movies
        self.item_similarity_matrix = cosine_similarity(movie_matrix)
        
        # Convert to DataFrame
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns
        )
        
        print(f"🎭 Calculated similarities for {len(self.item_similarity_matrix)} movies")
        
    def get_user_based_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations using user-based collaborative filtering.
        
        How it works:
        1. Find users similar to the target user
        2. Look at movies those similar users rated highly
        3. Recommend movies the target user hasn't seen yet
        
        Args:
            user_id: ID of user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call fit() first.")
            
        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"User {user_id} not found in training data!")
            
        print(f"🔍 Finding recommendations for user {user_id}...")
        
        # Get this user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Find similar users (exclude the user themselves)
        user_similarities = self.user_similarity_matrix.loc[user_id].drop(user_id)
        
        # Get top 20 most similar users
        similar_users = user_similarities.nlargest(20).index
        
        print(f"👥 Found {len(similar_users)} similar users")
        
        # Calculate weighted average ratings for movies
        recommendations = {}
        
        for movie_id in self.user_movie_matrix.columns:
            # Skip if user already rated this movie
            if user_ratings[movie_id] > 0:
                continue
                
            # Calculate predicted rating based on similar users
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user in similar_users:
                similarity = user_similarities[similar_user]
                rating = self.user_movie_matrix.loc[similar_user, movie_id]
                
                if rating > 0:  # Only consider if similar user rated this movie
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
                    
            # Calculate predicted rating
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations[movie_id] = predicted_rating
                
        # Sort recommendations by predicted rating
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        # Format results with movie information
        results = []
        for movie_id, score in sorted_recommendations:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            results.append({
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'predicted_rating': round(score, 2),
                'recommendation_type': 'user_based'
            })
            
        print(f"✅ Generated {len(results)} user-based recommendations")
        return results
        
    def get_item_based_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations using item-based collaborative filtering.
        
        How it works:
        1. Look at movies the user has rated highly
        2. Find movies similar to those
        3. Recommend the most similar movies they haven't seen
        
        Args:
            user_id: ID of user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended movies with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call fit() first.")
            
        if user_id not in self.user_movie_matrix.index:
            raise ValueError(f"User {user_id} not found in training data!")
            
        print(f"🎬 Finding item-based recommendations for user {user_id}...")
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Find movies user rated highly (rating >= 4)
        liked_movies = user_ratings[user_ratings >= 4.0].index
        
        print(f"💝 User liked {len(liked_movies)} movies")
        
        # Calculate recommendation scores for each movie
        recommendations = {}
        
        for movie_id in self.user_movie_matrix.columns:
            # Skip if user already rated this movie
            if user_ratings[movie_id] > 0:
                continue
                
            # Calculate similarity to movies user liked
            similarity_scores = []
            
            for liked_movie in liked_movies:
                similarity = self.item_similarity_matrix.loc[movie_id, liked_movie]
                similarity_scores.append(similarity)
                
            if similarity_scores:
                # Average similarity to all liked movies
                avg_similarity = np.mean(similarity_scores)
                recommendations[movie_id] = avg_similarity
                
        # Sort recommendations by similarity score
        sorted_recommendations = sorted(
            recommendations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_recommendations]
        
        # Format results
        results = []
        for movie_id, score in sorted_recommendations:
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
            results.append({
                'movie_id': int(movie_id),
                'title': movie_info['title'],
                'genres': movie_info['genres'],
                'similarity_score': round(score, 3),
                'recommendation_type': 'item_based'
            })
            
        print(f"✅ Generated {len(results)} item-based recommendations")
        return results
        
    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 10) -> List[Dict]:
        """
        Get recommendations using a hybrid approach.
        
        Combines user-based and item-based recommendations for better results.
        
        Why hybrid?
        - User-based is good for discovering new types of movies
        - Item-based is good for finding movies similar to what you like
        - Together they cover more ground!
        """
        print(f"🔀 Creating hybrid recommendations for user {user_id}...")
        
        # Get both types of recommendations
        user_based = self.get_user_based_recommendations(user_id, n_recommendations)
        item_based = self.get_item_based_recommendations(user_id, n_recommendations)
        
        # Combine and mix them (alternating)
        hybrid_results = []
        max_len = max(len(user_based), len(item_based))
        
        for i in range(max_len):
            # Add user-based recommendation if available
            if i < len(user_based):
                rec = user_based[i].copy()
                rec['recommendation_type'] = 'hybrid (user-based)'
                hybrid_results.append(rec)
                
            # Add item-based recommendation if available and not duplicate
            if i < len(item_based):
                item_rec = item_based[i]
                # Check if this movie is already in results
                if not any(r['movie_id'] == item_rec['movie_id'] for r in hybrid_results):
                    rec = item_rec.copy()
                    rec['recommendation_type'] = 'hybrid (item-based)'
                    hybrid_results.append(rec)
                    
            # Stop if we have enough recommendations
            if len(hybrid_results) >= n_recommendations:
                break
                
        return hybrid_results[:n_recommendations]
        
    def explain_recommendation(self, user_id: int, movie_id: int) -> str:
        """
        Explain why a movie was recommended to a user.
        
        This helps users understand and trust the recommendations!
        """
        if not self.is_trained:
            return "Cannot explain: model not trained"
            
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_info.empty:
            return "Movie not found"
            
        movie_title = movie_info.iloc[0]['title']
        
        # Find similar users who liked this movie
        user_similarities = self.user_similarity_matrix.loc[user_id]
        movie_ratings = self.user_movie_matrix[movie_id]
        
        # Users who rated this movie highly (4+ stars)
        fans = movie_ratings[movie_ratings >= 4.0].index
        
        # Similar users who are also fans
        similar_fans = []
        for fan in fans:
            if fan != user_id and fan in user_similarities.index:
                similarity = user_similarities[fan]
                if similarity > 0.1:  # At least 10% similar
                    similar_fans.append((fan, similarity))
                    
        if similar_fans:
            similar_fans.sort(key=lambda x: x[1], reverse=True)
            top_similar_fan = similar_fans[0]
            
            explanation = f"💡 We recommend '{movie_title}' because users similar to you (like user {top_similar_fan[0]}) gave it high ratings. "
            explanation += f"You have {top_similar_fan[1]:.1%} similarity with users who loved this movie!"
        else:
            explanation = f"💡 We recommend '{movie_title}' based on your movie preferences and viewing patterns."
            
        return explanation
        
    def evaluate_model(self, test_ratings: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate how well our recommendation system works.
        
        We'll use RMSE (Root Mean Square Error) to measure prediction accuracy.
        Lower RMSE = better predictions!
        """
        if not self.is_trained:
            raise ValueError("Model not trained! Call fit() first.")
            
        print("📊 Evaluating model performance...")
        
        predictions = []
        actuals = []
        
        for _, row in test_ratings.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                # Try to predict this rating
                predicted_rating = self._predict_rating(user_id, movie_id)
                if predicted_rating is not None:
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
            except:
                continue  # Skip if we can't predict
                
        if len(predictions) > 0:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
            
            print(f"✅ Evaluated on {len(predictions)} predictions")
            print(f"📈 RMSE: {rmse:.3f} (lower is better)")
            print(f"📈 MAE: {mae:.3f} (lower is better)")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'n_predictions': len(predictions)
            }
        else:
            print("❌ Could not evaluate - no valid predictions")
            return {'rmse': float('inf'), 'mae': float('inf'), 'n_predictions': 0}
            
    def _predict_rating(self, user_id: int, movie_id: int) -> float:
        """
        Predict what rating a user would give to a movie.
        
        This is the core prediction function used in evaluation.
        """
        if user_id not in self.user_movie_matrix.index:
            return None
        if movie_id not in self.user_movie_matrix.columns:
            return None
            
        # Get similar users
        user_similarities = self.user_similarity_matrix.loc[user_id]
        
        weighted_sum = 0
        similarity_sum = 0
        
        for other_user in user_similarities.index:
            if other_user == user_id:
                continue
                
            similarity = user_similarities[other_user]
            rating = self.user_movie_matrix.loc[other_user, movie_id]
            
            if rating > 0 and similarity > 0:
                weighted_sum += similarity * rating
                similarity_sum += similarity
                
        if similarity_sum > 0:
            return weighted_sum / similarity_sum
        else:
            # Fall back to average rating for this movie
            movie_ratings = self.user_movie_matrix[movie_id]
            non_zero_ratings = movie_ratings[movie_ratings > 0]
            if len(non_zero_ratings) > 0:
                return non_zero_ratings.mean()
            else:
                return 3.0  # Default neutral rating
                

# Let's create a simple test function
def test_recommender():
    """
    A simple function to test our recommendation system.
    """
    print("🧪 Testing our recommendation system...")
    
    # This would typically use real data
    # For now, let's just verify the class can be instantiated
    recommender = CollaborativeFilteringRecommender()
    print("✅ Recommender created successfully!")
    print("📚 Ready to train on real data!")


if __name__ == "__main__":
    test_recommender() 