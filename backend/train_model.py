"""
🎓 Model Training Script

This script demonstrates the complete machine learning pipeline for our recommendation system.
Perfect for beginners to understand how everything fits together!

What we'll do:
1. Load and explore the data
2. Split data into training and testing sets  
3. Train our recommendation model
4. Evaluate how well it works
5. Generate sample recommendations
6. Save the trained model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

# Import our custom modules
from data.data_loader import MovieDataLoader, explore_data_basics
from models.recommender import CollaborativeFilteringRecommender


def main():
    """
    Main training pipeline - this is where all the magic happens!
    """
    print("🎬" + "="*60)
    print("    MOVIE RECOMMENDATION SYSTEM TRAINING")
    print("="*60 + "🎬")
    
    # Step 1: Load and Explore Data
    print("\n📚 STEP 1: Loading Movie Data")
    print("-" * 40)
    
    # Create data loader and download dataset
    loader = MovieDataLoader(data_dir="../data")
    
    if not loader.download_dataset():
        print("❌ Failed to download dataset. Please check your internet connection!")
        return
    
    # Load sample data for training (we'll use a manageable size for learning)
    print("\n🎯 Loading sample data for training...")
    ratings_df, movies_df, user_movie_matrix = loader.get_sample_data(
        n_users=200,    # Use top 200 most active users
        n_movies=500    # Use top 500 most popular movies
    )
    
    # Explore the data to understand what we're working with
    explore_data_basics(ratings_df, movies_df)
    
    # Step 2: Prepare Data for Machine Learning
    print("\n🔄 STEP 2: Preparing Data for Training")
    print("-" * 40)
    
    # Split data into training and testing sets
    # Why do we split data?
    # - Training set: Used to teach our model
    # - Test set: Used to see how well our model works on unseen data
    # - This prevents overfitting (memorizing instead of learning patterns)
    
    print("📊 Splitting data into training (80%) and testing (20%) sets...")
    train_ratings, test_ratings = train_test_split(
        ratings_df, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducible results
        stratify=ratings_df['userId']  # Keep similar user distribution in both sets
    )
    
    print(f"✅ Training set: {len(train_ratings):,} ratings")
    print(f"✅ Testing set: {len(test_ratings):,} ratings")
    
    # Step 3: Train the Recommendation Model
    print("\n🏋️ STEP 3: Training Recommendation Model")
    print("-" * 40)
    
    # Create and train our recommender
    recommender = CollaborativeFilteringRecommender(min_ratings=5)
    
    print("🤖 Training collaborative filtering model...")
    print("This might take a minute - we're calculating similarities for all users and movies!")
    
    # Train the model on our training data
    recommender.fit(train_ratings, movies_df)
    
    # Step 4: Evaluate Model Performance
    print("\n📊 STEP 4: Evaluating Model Performance")
    print("-" * 40)
    
    # Test how well our model predicts ratings
    evaluation_results = recommender.evaluate_model(test_ratings)
    
    print("\n📈 Model Performance Results:")
    print(f"   RMSE: {evaluation_results['rmse']:.3f}")
    print(f"   MAE: {evaluation_results['mae']:.3f}")
    print(f"   Predictions made: {evaluation_results['n_predictions']:,}")
    
    # What do these numbers mean?
    print("\n💡 What these metrics mean:")
    print("   📌 RMSE (Root Mean Square Error): How far off our predictions are on average")
    print("      - Lower is better (0 = perfect predictions)")
    print("      - Good RMSE for movie ratings is typically < 1.0")
    print("   📌 MAE (Mean Absolute Error): Average absolute difference between predicted and actual ratings")
    print("      - Also lower is better")
    
    # Step 5: Generate Sample Recommendations
    print("\n🎁 STEP 5: Generating Sample Recommendations")
    print("-" * 40)
    
    # Let's test recommendations for a few users
    sample_users = train_ratings['userId'].unique()[:3]  # Get first 3 users
    
    for user_id in sample_users:
        print(f"\n👤 Recommendations for User {user_id}:")
        print("   🔍 User-based recommendations:")
        
        try:
            # Get user-based recommendations
            user_recs = recommender.get_user_based_recommendations(user_id, n_recommendations=5)
            for i, rec in enumerate(user_recs, 1):
                print(f"      {i}. {rec['title']} (Predicted rating: {rec['predicted_rating']:.1f})")
                
        except Exception as e:
            print(f"      ❌ Could not generate recommendations: {e}")
            
        print("   🎬 Item-based recommendations:")
        try:
            # Get item-based recommendations
            item_recs = recommender.get_item_based_recommendations(user_id, n_recommendations=5)
            for i, rec in enumerate(item_recs, 1):
                print(f"      {i}. {rec['title']} (Similarity: {rec['similarity_score']:.3f})")
                
        except Exception as e:
            print(f"      ❌ Could not generate recommendations: {e}")
    
    # Step 6: Demonstrate Recommendation Explanations
    print("\n💡 STEP 6: Explaining Recommendations")
    print("-" * 40)
    
    # Show how we can explain why movies were recommended
    try:
        user_id = sample_users[0]
        recommendations = recommender.get_user_based_recommendations(user_id, n_recommendations=3)
        
        print(f"\n🗣️  Explanations for User {user_id}:")
        for rec in recommendations:
            movie_id = rec['movie_id']
            explanation = recommender.explain_recommendation(user_id, movie_id)
            print(f"   🎬 {rec['title']}")
            print(f"      {explanation}")
            print()
            
    except Exception as e:
        print(f"❌ Could not generate explanations: {e}")
    
    # Step 7: Save the Trained Model
    print("\n💾 STEP 7: Saving Trained Model")
    print("-" * 40)
    
    # Create models directory if it doesn't exist
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    # Save the trained model
    model_path = models_dir / "collaborative_filtering_model.pkl"
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        print(f"✅ Model saved to: {model_path}")
        
        # Also save the movies data for later use
        movies_path = models_dir / "movies_data.pkl"
        with open(movies_path, 'wb') as f:
            pickle.dump(movies_df, f)
        print(f"✅ Movies data saved to: {movies_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
    
    # Training Complete!
    print("\n🎉" + "="*60)
    print("    TRAINING COMPLETE!")
    print("="*60 + "🎉")
    
    print("\n📋 Summary:")
    print(f"   ✅ Trained on {len(train_ratings):,} ratings")
    print(f"   ✅ Tested on {len(test_ratings):,} ratings")
    print(f"   ✅ Model RMSE: {evaluation_results['rmse']:.3f}")
    print(f"   ✅ Model saved and ready to use!")
    
    print("\n🚀 Next steps:")
    print("   1. Create a FastAPI backend to serve recommendations")
    print("   2. Build a React frontend for user interaction")
    print("   3. Add AI explanations using LangChain")
    print("   4. Deploy your recommendation system!")
    
    return recommender, evaluation_results


def analyze_recommendations_quality(recommender, movies_df, sample_users):
    """
    Analyze the quality and diversity of recommendations.
    
    This helps us understand if our recommendations are good and diverse.
    """
    print("\n🔍 ANALYZING RECOMMENDATION QUALITY")
    print("-" * 40)
    
    all_recommendations = []
    genres_recommended = []
    
    for user_id in sample_users[:10]:  # Analyze first 10 users
        try:
            recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
            all_recommendations.extend(recs)
            
            # Collect genres
            for rec in recs:
                genres = rec['genres'].split('|')
                genres_recommended.extend(genres)
                
        except:
            continue
    
    if all_recommendations:
        # Analyze recommendation diversity
        unique_movies = len(set(rec['movie_id'] for rec in all_recommendations))
        total_recommendations = len(all_recommendations)
        
        print(f"📊 Recommendation Diversity:")
        print(f"   Total recommendations made: {total_recommendations}")
        print(f"   Unique movies recommended: {unique_movies}")
        print(f"   Diversity score: {unique_movies/total_recommendations:.2%}")
        
        # Analyze genre diversity
        from collections import Counter
        genre_counts = Counter(genres_recommended)
        top_genres = genre_counts.most_common(5)
        
        print(f"\n🎭 Top Recommended Genres:")
        for genre, count in top_genres:
            print(f"   {genre}: {count} recommendations")
    
    print("\n💡 Quality Insights:")
    print("   📌 Good diversity score: > 50% (many different movies recommended)")
    print("   📌 Balanced genres: No single genre dominates recommendations")
    print("   📌 Popular vs Niche: Mix of popular and lesser-known movies is ideal")


def create_simple_demo():
    """
    Create a simple interactive demo for testing recommendations.
    """
    print("\n🎮 INTERACTIVE DEMO MODE")
    print("-" * 40)
    print("This is where you could add interactive features like:")
    print("   1. Ask user to rate some movies")
    print("   2. Generate personalized recommendations")
    print("   3. Show explanations for each recommendation")
    print("   4. Allow user to feedback on recommendations")
    print("\nFor now, this is a placeholder for future interactive features!")


if __name__ == "__main__":
    """
    Run the complete training pipeline when this script is executed.
    """
    try:
        # Run the main training pipeline
        recommender, results = main()
        
        # Optional: Run additional analysis
        if recommender.is_trained:
            # Load some sample data for analysis
            loader = MovieDataLoader(data_dir="../data")
            ratings_df, movies_df, _ = loader.get_sample_data(n_users=50, n_movies=200)
            sample_users = ratings_df['userId'].unique()
            
            # Analyze recommendation quality
            analyze_recommendations_quality(recommender, movies_df, sample_users)
            
            # Show demo mode
            create_simple_demo()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n👋 Thanks for training with us! Check out the saved model and start building your API!") 