"""
🚀 FastAPI Backend for Movie Recommendation System

This is the main API server that connects our machine learning model
to the web interface. It's built with FastAPI for speed and automatic documentation.

What this API does:
- Serves movie recommendations from our trained model
- Handles user ratings and preferences
- Provides movie information and explanations
- Manages user sessions
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import uvicorn
from contextlib import asynccontextmanager

# Import our recommendation system
from models.recommender import CollaborativeFilteringRecommender
from data.data_loader import MovieDataLoader

# Global variables for our model and data
recommender = None
movies_df = None
original_ratings_df = None

async def load_model():
    """
    Load our trained model when the server starts.
    
    This runs once when the API starts up and loads everything into memory
    for fast responses to user requests.
    """
    global recommender, movies_df, original_ratings_df
    
    print("🚀 Starting Movie Recommendation API...")
    
    try:
        # Load the trained model
        models_dir = Path("../models")
        model_path = models_dir / "collaborative_filtering_model.pkl"
        movies_path = models_dir / "movies_data.pkl"
        
        if model_path.exists() and movies_path.exists():
            print("📦 Loading trained model...")
            with open(model_path, 'rb') as f:
                recommender = pickle.load(f)
                
            with open(movies_path, 'rb') as f:
                movies_df = pickle.load(f)
                
            print("✅ Model loaded successfully!")
            print(f"📊 Model knows about {len(movies_df)} movies")
            
            # Also load original ratings for reference
            loader = MovieDataLoader(data_dir="../data")
            original_ratings_df, _, _ = loader.get_sample_data(n_users=200, n_movies=500)
            
        else:
            print("❌ Trained model not found! Please run train_model.py first.")
            print("🔧 Creating a dummy model for development...")
            
            # Create dummy data for development
            loader = MovieDataLoader(data_dir="../data")
            if loader.download_dataset():
                original_ratings_df, movies_df, _ = loader.get_sample_data(n_users=50, n_movies=100)
                recommender = CollaborativeFilteringRecommender()
                recommender.fit(original_ratings_df, movies_df)
                print("✅ Dummy model created for development!")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔧 API will run in limited mode without recommendations")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown (if needed)

# Initialize FastAPI app
app = FastAPI(
    title="🎬 Movie Recommendation API",
    description="AI-powered movie recommendations with explanations",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc", # ReDoc at /redoc
    lifespan=lifespan
)

# Allow requests from frontend (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React default
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class MovieRating(BaseModel):
    """Model for when a user rates a movie"""
    user_id: int
    movie_id: int
    rating: float
    
class RecommendationRequest(BaseModel):
    """Model for requesting recommendations"""
    user_id: int
    num_recommendations: int = 10
    recommendation_type: str = "hybrid"  # "user", "item", or "hybrid"

class MovieInfo(BaseModel):
    """Model for movie information"""
    movie_id: int
    title: str
    genres: str
    
class RecommendationResponse(BaseModel):
    """Model for recommendation responses"""
    movie_id: int
    title: str
    genres: str
    score: float
    recommendation_type: str
    explanation: Optional[str] = None


@app.get("/")
async def root():
    """
    Welcome endpoint - shows basic API information.
    """
    return {
        "message": "🎬 Welcome to the Movie Recommendation API!",
        "status": "running",
        "model_loaded": recommender is not None,
        "total_movies": len(movies_df) if movies_df is not None else 0,
        "docs": "/docs",
        "endpoints": {
            "movies": "/movies",
            "recommendations": "/recommendations",
            "rate_movie": "/rate",
            "movie_info": "/movies/{movie_id}",
            "explain": "/explain/{user_id}/{movie_id}"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint - useful for monitoring.
    """
    return {
        "status": "healthy",
        "model_loaded": recommender is not None and recommender.is_trained,
        "movies_available": movies_df is not None
    }


@app.get("/movies", response_model=List[MovieInfo])
async def get_movies(
    limit: int = Query(50, description="Number of movies to return", ge=1, le=500),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    search: Optional[str] = Query(None, description="Search in movie titles")
):
    """
    Get a list of available movies with optional filtering.
    
    Args:
        limit: Maximum number of movies to return
        genre: Filter movies by genre (e.g., "Action", "Comedy")
        search: Search for movies containing this text in the title
    
    Returns:
        List of movies with their information
    """
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Movie data not available")
    
    # Start with all movies
    filtered_movies = movies_df.copy()
    
    # Apply genre filter
    if genre:
        filtered_movies = filtered_movies[
            filtered_movies['genres'].str.contains(genre, case=False, na=False)
        ]
    
    # Apply search filter
    if search:
        filtered_movies = filtered_movies[
            filtered_movies['title'].str.contains(search, case=False, na=False)
        ]
    
    # Limit results
    filtered_movies = filtered_movies.head(limit)
    
    # Convert to response format
    result = []
    for _, movie in filtered_movies.iterrows():
        result.append(MovieInfo(
            movie_id=int(movie['movieId']),
            title=movie['title'],
            genres=movie['genres']
        ))
    
    return result


@app.get("/movies/{movie_id}")
async def get_movie(movie_id: int):
    """
    Get detailed information about a specific movie.
    
    Args:
        movie_id: The ID of the movie
    
    Returns:
        Movie information with additional details
    """
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Movie data not available")
    
    movie = movies_df[movies_df['movieId'] == movie_id]
    
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_data = movie.iloc[0]
    
    # Calculate additional statistics if we have ratings data
    additional_info = {}
    if original_ratings_df is not None:
        movie_ratings = original_ratings_df[original_ratings_df['movieId'] == movie_id]
        if not movie_ratings.empty:
            additional_info = {
                "average_rating": round(movie_ratings['rating'].mean(), 2),
                "total_ratings": len(movie_ratings),
                "rating_distribution": movie_ratings['rating'].value_counts().to_dict()
            }
    
    return {
        "movie_id": int(movie_data['movieId']),
        "title": movie_data['title'],
        "genres": movie_data['genres'],
        **additional_info
    }


@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized movie recommendations for a user.
    
    This is the main recommendation endpoint that uses our trained ML model
    to suggest movies the user might like.
    
    Args:
        request: Contains user_id, number of recommendations, and type
        
    Returns:
        List of recommended movies with scores and explanations
    """
    if recommender is None or not recommender.is_trained:
        raise HTTPException(status_code=503, detail="Recommendation model not available")
    
    try:
        # Generate recommendations based on type
        if request.recommendation_type == "user":
            recommendations = recommender.get_user_based_recommendations(
                request.user_id, request.num_recommendations
            )
        elif request.recommendation_type == "item":
            recommendations = recommender.get_item_based_recommendations(
                request.user_id, request.num_recommendations
            )
        else:  # hybrid
            recommendations = recommender.get_hybrid_recommendations(
                request.user_id, request.num_recommendations
            )
        
        # Format response with explanations
        result = []
        for rec in recommendations:
            # Get explanation for this recommendation
            explanation = recommender.explain_recommendation(
                request.user_id, rec['movie_id']
            )
            
            # Determine score field name based on recommendation type
            score_field = 'predicted_rating' if 'predicted_rating' in rec else 'similarity_score'
            score = rec.get(score_field, 0.0)
            
            result.append(RecommendationResponse(
                movie_id=rec['movie_id'],
                title=rec['title'],
                genres=rec['genres'],
                score=score,
                recommendation_type=rec['recommendation_type'],
                explanation=explanation
            ))
        
        return {
            "user_id": request.user_id,
            "recommendations": result,
            "total_count": len(result)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.post("/rate")
async def rate_movie(rating: MovieRating):
    """
    Submit a rating for a movie.
    
    In a real system, this would update the model with new user preferences.
    For now, it just stores the rating for future model updates.
    
    Args:
        rating: User rating for a movie (user_id, movie_id, rating)
        
    Returns:
        Confirmation of the rating submission
    """
    # Validate rating value
    if not 0.5 <= rating.rating <= 5.0:
        raise HTTPException(
            status_code=400, 
            detail="Rating must be between 0.5 and 5.0"
        )
    
    # Validate movie exists
    if movies_df is not None:
        movie_exists = not movies_df[movies_df['movieId'] == rating.movie_id].empty
        if not movie_exists:
            raise HTTPException(status_code=404, detail="Movie not found")
    
    # In a real system, you'd save this to a database
    # For now, we'll just acknowledge the rating
    
    return {
        "message": "Rating submitted successfully!",
        "user_id": rating.user_id,
        "movie_id": rating.movie_id,
        "rating": rating.rating,
        "note": "In a production system, this would update the recommendation model"
    }


@app.get("/explain/{user_id}/{movie_id}")
async def explain_recommendation(user_id: int, movie_id: int):
    """
    Get an explanation for why a movie was recommended to a user.
    
    This helps users understand the reasoning behind recommendations,
    which increases trust and engagement.
    
    Args:
        user_id: The user who received the recommendation
        movie_id: The movie that was recommended
        
    Returns:
        Detailed explanation of the recommendation
    """
    if recommender is None or not recommender.is_trained:
        raise HTTPException(status_code=503, detail="Recommendation model not available")
    
    try:
        explanation = recommender.explain_recommendation(user_id, movie_id)
        
        # Get movie information
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        if movie_info.empty:
            raise HTTPException(status_code=404, detail="Movie not found")
        
        movie_data = movie_info.iloc[0]
        
        return {
            "user_id": user_id,
            "movie_id": movie_id,
            "movie_title": movie_data['title'],
            "explanation": explanation,
            "technical_note": "This explanation is based on collaborative filtering similarity scores"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """
    Get a user's viewing profile and preferences.
    
    This shows what genres they like, how many movies they've rated, etc.
    
    Args:
        user_id: The user's ID
        
    Returns:
        User profile information
    """
    if original_ratings_df is None:
        raise HTTPException(status_code=503, detail="User data not available")
    
    user_ratings = original_ratings_df[original_ratings_df['userId'] == user_id]
    
    if user_ratings.empty:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Calculate user statistics
    total_ratings = len(user_ratings)
    average_rating = user_ratings['rating'].mean()
    
    # Get genre preferences
    user_movies = user_ratings.merge(movies_df, on='movieId')
    genre_ratings = []
    
    # Calculate average rating per genre
    for _, movie in user_movies.iterrows():
        genres = movie['genres'].split('|')
        for genre in genres:
            genre_ratings.append({
                'genre': genre.strip(),
                'rating': movie['rating']
            })
    
    genre_df = pd.DataFrame(genre_ratings)
    if not genre_df.empty:
        genre_preferences = genre_df.groupby('genre')['rating'].agg(['mean', 'count']).round(2)
        top_genres = genre_preferences.sort_values('mean', ascending=False).head(5)
    else:
        top_genres = pd.DataFrame()
    
    return {
        "user_id": user_id,
        "total_ratings": total_ratings,
        "average_rating": round(average_rating, 2),
        "top_genres": top_genres.to_dict('index') if not top_genres.empty else {},
        "rating_distribution": user_ratings['rating'].value_counts().to_dict()
    }


# Run the server
if __name__ == "__main__":
    print("🎬 Starting Movie Recommendation API Server...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔧 Alternative documentation at: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes during development
        log_level="info"
    ) 