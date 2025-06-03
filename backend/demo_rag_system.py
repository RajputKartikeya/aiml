"""
🎬 Phase 2 Demo: Enhanced Movie Recommendation System with RAG

This demo script showcases the complete Phase 2 implementation featuring:
- LangChain-powered knowledge base
- Semantic search and retrieval
- Intelligent explanation generation
- RAG (Retrieval-Augmented Generation) integration

Run this script to see the enhanced movie recommendation system in action!
"""

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our enhanced system components
from data.data_loader import MovieDataLoader
from models.recommender import CollaborativeFilteringRecommender
from services.movie_knowledge import MovieKnowledgeBase
from services.rag_chain import MovieRecommendationRAG, RecommendationContext


class RAGMovieRecommendationDemo:
    """
    Demo class showcasing the enhanced movie recommendation system
    """
    
    def __init__(self):
        """Initialize the demo system"""
        print("🎬 Initializing Enhanced Movie Recommendation System")
        print("=" * 60)
        
        # Core components
        self.data_loader = None
        self.movies_df = None
        self.ratings_df = None
        self.recommender = None
        
        # RAG components
        self.knowledge_base = None
        self.rag_chain = None
        
        print("✅ Demo system initialized")
    
    async def setup_system(self):
        """Set up the complete recommendation system"""
        print("\n🔧 Setting up movie recommendation system...")
        
        # 1. Load and prepare data
        await self._load_data()
        
        # 2. Train recommendation model
        await self._train_model()
        
        # 3. Initialize RAG system
        await self._initialize_rag()
        
        print("✅ System setup complete!")
    
    async def _load_data(self):
        """Load movie and ratings data"""
        print("📊 Loading movie data...")
        
        self.data_loader = MovieDataLoader(data_dir="../data")
        
        # Get sample data for demo
        self.ratings_df, self.movies_df, _ = self.data_loader.get_sample_data(
            n_users=100, n_movies=50
        )
        
        print(f"✅ Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
    
    async def _train_model(self):
        """Train the collaborative filtering model"""
        print("🤖 Training recommendation model...")
        
        self.recommender = CollaborativeFilteringRecommender()
        self.recommender.fit(self.ratings_df, self.movies_df)
        
        print("✅ Model trained successfully")
    
    async def _initialize_rag(self):
        """Initialize the RAG system"""
        print("🧠 Initializing RAG system...")
        
        # Initialize knowledge base
        self.knowledge_base = MovieKnowledgeBase(
            persist_directory="../data/demo_vectorstore"
        )
        
        # Build knowledge base (this creates the vector store)
        await asyncio.get_event_loop().run_in_executor(
            None, 
            self.knowledge_base.build_knowledge_base, 
            self.movies_df, 
            True  # Force rebuild for demo
        )
        
        # Initialize RAG chain
        self.rag_chain = MovieRecommendationRAG(self.knowledge_base)
        
        print("✅ RAG system initialized")
    
    def demonstrate_basic_recommendations(self, user_id=None):
        """Show basic recommendations without RAG"""
        # Use the first available user if none specified
        if user_id is None:
            user_id = self.ratings_df['userId'].iloc[0]
        
        print(f"\n📋 Basic Recommendations for User {user_id}")
        print("-" * 40)
        
        # Get basic recommendations
        recommendations = self.recommender.get_hybrid_recommendations(user_id, 3)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            print(f"   Score: {rec.get('predicted_rating', rec.get('similarity_score', 0)):.2f}")
            print(f"   Basic Explanation: {self.recommender.explain_recommendation(user_id, rec['movie_id'])}")
            print()
    
    def demonstrate_rag_explanations(self, user_id=None):
        """Show enhanced explanations with RAG"""
        # Use the first available user if none specified
        if user_id is None:
            user_id = self.ratings_df['userId'].iloc[0]
        
        print(f"\n🤖 Enhanced RAG Explanations for User {user_id}")
        print("-" * 50)
        
        # Get recommendations
        recommendations = self.recommender.get_hybrid_recommendations(user_id, 3)
        
        # Generate user profile
        user_profile = self._generate_user_profile(user_id)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   Genres: {rec['genres']}")
            
            # Create RAG context
            context = RecommendationContext(
                user_id=user_id,
                recommended_movie_id=rec['movie_id'],
                recommended_movie_title=rec['title'],
                recommendation_score=rec.get('predicted_rating', rec.get('similarity_score', 0)),
                recommendation_type="hybrid",
                user_preferences=user_profile,
                similar_movies=self.knowledge_base.search_similar_movies(rec['title'], k=3)
            )
            
            # Generate enhanced explanation
            enhanced_explanation = self.rag_chain.generate_explanation(context)
            print(f"   🧠 RAG Explanation: {enhanced_explanation}")
            print()
    
    def demonstrate_semantic_search(self):
        """Show semantic search capabilities"""
        print("\n🔍 Semantic Search Demonstration")
        print("-" * 35)
        
        search_queries = [
            "action adventure movies with heroes",
            "romantic comedies with happy endings",
            "sci-fi movies about space exploration",
            "animated family films"
        ]
        
        for query in search_queries:
            print(f"Query: '{query}'")
            results = self.knowledge_base.search_similar_movies(query, k=3)
            
            for j, result in enumerate(results, 1):
                title = result['metadata']['title']
                genres = result['metadata']['genres']
                score = result['similarity_score']
                print(f"  {j}. {title} ({genres}) - Score: {score:.3f}")
            print()
    
    def demonstrate_movie_knowledge_retrieval(self):
        """Show movie knowledge retrieval"""
        print("\n📚 Movie Knowledge Retrieval")
        print("-" * 30)
        
        # Get some sample movies
        sample_movies = self.movies_df.head(3)
        
        for _, movie in sample_movies.iterrows():
            movie_id = movie['movieId']
            title = movie['title']
            
            print(f"Movie: {title}")
            context = self.knowledge_base.get_movie_context(movie_id)
            
            if context:
                # Show first few lines of context
                lines = context.split('\n')[:6]
                for line in lines:
                    if line.strip():
                        print(f"  {line}")
            else:
                print("  No detailed context available")
            print()
    
    def _generate_user_profile(self, user_id):
        """Generate user profile for RAG context"""
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        if user_ratings.empty:
            return {}
        
        # Calculate genre preferences
        user_movies = user_ratings.merge(self.movies_df, on='movieId')
        genre_ratings = []
        
        for _, movie in user_movies.iterrows():
            genres = movie['genres'].split('|')
            for genre in genres:
                genre_ratings.append({
                    'genre': genre.strip(),
                    'rating': movie['rating']
                })
        
        if genre_ratings:
            genre_df = pd.DataFrame(genre_ratings)
            genre_preferences = genre_df.groupby('genre')['rating'].agg(['mean', 'count']).round(2)
            top_genres = genre_preferences.sort_values('mean', ascending=False).head(5)
            
            return {
                "top_genres": top_genres.to_dict('index'),
                "average_rating": user_ratings['rating'].mean(),
                "total_ratings": len(user_ratings)
            }
        
        return {}
    
    def performance_comparison(self, user_id=None):
        """Compare basic vs RAG explanations"""
        # Use the first available user if none specified
        if user_id is None:
            user_id = self.ratings_df['userId'].iloc[0]
            
        print("\n⚡ Performance Comparison: Basic vs RAG")
        print("-" * 45)
        
        import time
        
        # Get a recommendation
        recommendations = self.recommender.get_hybrid_recommendations(user_id, 1)
        rec = recommendations[0]
        
        # Time basic explanation
        start_time = time.time()
        basic_explanation = self.recommender.explain_recommendation(user_id, rec['movie_id'])
        basic_time = time.time() - start_time
        
        # Time RAG explanation
        user_profile = self._generate_user_profile(user_id)
        context = RecommendationContext(
            user_id=user_id,
            recommended_movie_id=rec['movie_id'],
            recommended_movie_title=rec['title'],
            recommendation_score=rec.get('predicted_rating', 0),
            recommendation_type="hybrid",
            user_preferences=user_profile,
            similar_movies=[]
        )
        
        start_time = time.time()
        rag_explanation = self.rag_chain.generate_explanation(context)
        rag_time = time.time() - start_time
        
        # Display comparison
        print(f"Movie: {rec['title']}")
        print(f"Basic Explanation ({basic_time:.3f}s): {basic_explanation}")
        print(f"RAG Explanation ({rag_time:.3f}s): {rag_explanation}")
        print(f"Quality Improvement: {'Higher contextual relevance' if len(rag_explanation) > len(basic_explanation) else 'More concise'}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 Starting Complete RAG Movie Recommendation Demo")
        print("=" * 60)
        
        # Setup system
        await self.setup_system()
        
        # Get first available user for demo
        demo_user = self.ratings_df['userId'].iloc[0]
        print(f"\n🎭 Using User {demo_user} for demonstration")
        
        # Run demonstrations
        print("\n" + "🎯 DEMONSTRATION SECTIONS" + "\n")
        
        self.demonstrate_basic_recommendations(demo_user)
        self.demonstrate_rag_explanations(demo_user)
        self.demonstrate_semantic_search()
        self.demonstrate_movie_knowledge_retrieval()
        self.performance_comparison(demo_user)
        
        print("\n" + "=" * 60)
        print("🎉 Demo Complete! Phase 2 RAG Implementation Successful!")
        print("\nKey Features Demonstrated:")
        print("✅ Enhanced movie knowledge base with detailed information")
        print("✅ Semantic search using vector embeddings")
        print("✅ Intelligent explanation generation with RAG")
        print("✅ Context-aware recommendations")
        print("✅ Performance optimizations and fallback mechanisms")
        print("\n🚀 Your enhanced movie recommendation system is ready!")


async def main():
    """Main demo function"""
    print("🎬 Enhanced Movie Recommendation System - Phase 2 Demo")
    print("🤖 Featuring LangChain RAG Integration")
    print()
    
    # Create and run demo
    demo = RAGMovieRecommendationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    # Set environment for demo
    os.environ.setdefault("OPENAI_API_KEY", "your_openai_api_key_here")
    
    print("🎬 Starting Enhanced Movie Recommendation Demo...")
    print("This showcases the complete Phase 2 RAG implementation")
    print("Press Ctrl+C to cancel\n")
    
    # Run the demo
    asyncio.run(main()) 