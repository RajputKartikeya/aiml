"""
🎬 Movie Data Loader

This module handles downloading and loading the MovieLens dataset.
It's designed to be beginner-friendly with lots of explanations!

What is MovieLens?
- MovieLens is a famous dataset used for recommendation systems
- It contains user ratings for movies, movie information, and user data
- Perfect for learning how recommendation systems work!
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


class MovieDataLoader:
    """
    A friendly class to help us load movie data for our recommendation system.
    
    Think of this as your data assistant - it knows how to:
    1. Download movie data from the internet
    2. Clean and organize the data
    3. Make it ready for our machine learning models
    """
    
    def __init__(self, data_dir: str = "../../data"):
        """
        Initialize our data loader.
        
        Args:
            data_dir: Where we'll store our movie data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)  # Create the directory if it doesn't exist
        
        # MovieLens dataset URLs (we'll use the small version for learning)
        self.dataset_url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        self.dataset_name = "ml-latest-small"
        
    def download_dataset(self) -> bool:
        """
        Downloads the MovieLens dataset if we don't have it already.
        
        Returns:
            True if download was successful, False otherwise
        """
        zip_path = self.data_dir / "movielens.zip"
        extract_path = self.data_dir / self.dataset_name
        
        # Check if we already have the data
        if extract_path.exists():
            print("✅ MovieLens dataset already exists! Skipping download.")
            return True
            
        try:
            print("📥 Downloading MovieLens dataset... This might take a minute!")
            
            # Download the dataset
            response = requests.get(self.dataset_url, stream=True)
            response.raise_for_status()
            
            # Save the zip file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            print("✅ Download complete! Now extracting...")
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            # Clean up the zip file
            zip_path.unlink()
            
            print("🎉 Dataset ready to use!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            return False
            
    def load_ratings(self) -> pd.DataFrame:
        """
        Load the ratings data - this is the heart of our recommendation system!
        
        What's in ratings data?
        - userId: Which user rated the movie
        - movieId: Which movie was rated
        - rating: How much they liked it (0.5 to 5.0 stars)
        - timestamp: When they rated it
        
        Returns:
            DataFrame with all user ratings
        """
        ratings_path = self.data_dir / self.dataset_name / "ratings.csv"
        
        if not ratings_path.exists():
            raise FileNotFoundError("Ratings file not found! Did you download the dataset?")
            
        print("📊 Loading ratings data...")
        ratings = pd.read_csv(ratings_path)
        
        # Let's explore the data a bit
        print(f"📈 Found {len(ratings):,} ratings from {ratings['userId'].nunique():,} users")
        print(f"🎬 They rated {ratings['movieId'].nunique():,} different movies")
        print(f"⭐ Average rating: {ratings['rating'].mean():.1f} stars")
        
        return ratings
        
    def load_movies(self) -> pd.DataFrame:
        """
        Load movie information - titles, genres, etc.
        
        What's in movies data?
        - movieId: Unique identifier for each movie
        - title: Movie name and year
        - genres: What types of movie it is (Action, Comedy, etc.)
        
        Returns:
            DataFrame with movie information
        """
        movies_path = self.data_dir / self.dataset_name / "movies.csv"
        
        if not movies_path.exists():
            raise FileNotFoundError("Movies file not found! Did you download the dataset?")
            
        print("🎬 Loading movie information...")
        movies = pd.read_csv(movies_path)
        
        print(f"🎭 Found information for {len(movies):,} movies")
        
        # Let's see what genres we have
        all_genres = movies['genres'].str.split('|').explode().unique()
        print(f"🎪 Available genres: {', '.join(all_genres[:10])}...")  # Show first 10
        
        return movies
        
    def create_user_movie_matrix(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create a user-movie matrix - this is crucial for collaborative filtering!
        
        What is a user-movie matrix?
        - Rows = users, Columns = movies
        - Each cell = rating that user gave to that movie
        - Empty cells = user hasn't rated that movie
        
        This matrix is the foundation of collaborative filtering because:
        1. We can find similar users (users who rated movies similarly)
        2. We can find similar movies (movies that got similar ratings)
        3. We can predict missing ratings!
        
        Args:
            ratings: DataFrame with user ratings
            
        Returns:
            DataFrame where rows=users, columns=movies, values=ratings
        """
        print("🔄 Creating user-movie matrix...")
        
        # This is like creating a giant spreadsheet:
        # - Each row is a user
        # - Each column is a movie  
        # - Each cell is the rating (if they rated it)
        user_movie_matrix = ratings.pivot_table(
            index='userId',      # Rows will be users
            columns='movieId',   # Columns will be movies
            values='rating',     # Fill cells with ratings
            fill_value=0         # If no rating, put 0
        )
        
        print(f"📊 Matrix shape: {user_movie_matrix.shape} (users × movies)")
        print(f"🕳️  Sparsity: {(user_movie_matrix == 0).sum().sum() / user_movie_matrix.size * 100:.1f}% empty")
        
        return user_movie_matrix
        
    def get_sample_data(self, n_users: int = 100, n_movies: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get a smaller sample of data for testing and development.
        
        Why use a sample?
        - Full dataset can be slow for development
        - Easier to understand and debug with smaller data
        - Still representative of the real problem
        
        Args:
            n_users: Number of users to include
            n_movies: Number of movies to include
            
        Returns:
            Tuple of (ratings_sample, movies_sample, user_movie_matrix_sample)
        """
        print(f"🎯 Creating sample with {n_users} users and {n_movies} movies...")
        
        # Load full data
        ratings = self.load_ratings()
        movies = self.load_movies()
        
        # Get most active users (users who rated the most movies)
        user_counts = ratings.groupby('userId').size()
        top_users = user_counts.nlargest(n_users).index
        
        # Get most popular movies (movies with most ratings)
        movie_counts = ratings.groupby('movieId').size()
        popular_movies = movie_counts.nlargest(n_movies).index
        
        # Filter ratings to only include our selected users and movies
        sample_ratings = ratings[
            (ratings['userId'].isin(top_users)) & 
            (ratings['movieId'].isin(popular_movies))
        ]
        
        # Filter movies to only include those in our sample
        sample_movies = movies[movies['movieId'].isin(popular_movies)]
        
        # Create the user-movie matrix for our sample
        sample_matrix = self.create_user_movie_matrix(sample_ratings)
        
        print(f"✅ Sample ready! {len(sample_ratings):,} ratings, {len(sample_movies):,} movies")
        
        return sample_ratings, sample_movies, sample_matrix


def explore_data_basics(ratings: pd.DataFrame, movies: pd.DataFrame):
    """
    A helpful function to explore our data and understand what we're working with.
    
    This is important because understanding your data is the first step
    in any machine learning project!
    """
    print("\n" + "="*50)
    print("🔍 DATA EXPLORATION TIME!")
    print("="*50)
    
    # Rating distribution
    print("\n📊 RATING DISTRIBUTION:")
    rating_counts = ratings['rating'].value_counts().sort_index()
    for rating, count in rating_counts.items():
        stars = "⭐" * int(rating)
        print(f"{stars} {rating}: {count:,} ratings")
        
    # Most rated movies
    print("\n🏆 MOST POPULAR MOVIES:")
    movie_ratings = ratings.groupby('movieId').size().sort_values(ascending=False)
    top_movies = movie_ratings.head().index
    
    for movie_id in top_movies:
        movie_title = movies[movies['movieId'] == movie_id]['title'].iloc[0]
        count = movie_ratings[movie_id]
        print(f"🎬 {movie_title}: {count:,} ratings")
        
    # Most active users
    print("\n👥 MOST ACTIVE USERS:")
    user_ratings = ratings.groupby('userId').size().sort_values(ascending=False)
    print(f"🏅 Most active user rated {user_ratings.iloc[0]:,} movies!")
    print(f"📈 Average user rated {user_ratings.mean():.0f} movies")
    
    print("\n" + "="*50)


# Let's create a simple script to test our data loader
if __name__ == "__main__":
    print("🎬 Testing our Movie Data Loader!")
    
    # Create data loader
    loader = MovieDataLoader()
    
    # Download dataset
    if loader.download_dataset():
        # Load and explore data
        ratings, movies, matrix = loader.get_sample_data(n_users=50, n_movies=200)
        explore_data_basics(ratings, movies)
    else:
        print("❌ Failed to download dataset. Check your internet connection!") 