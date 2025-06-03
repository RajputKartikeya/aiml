"""
🧪 Test Script for RAG System

This script tests our enhanced movie recommendation system with LangChain/RAG
to ensure everything works correctly before integration.

Usage: python test_rag_system.py
"""

import os
import sys
import asyncio
import pandas as pd
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our RAG components
from data.data_loader import MovieDataLoader
from services.movie_knowledge import MovieKnowledgeBase
from services.rag_chain import MovieRecommendationRAG, RecommendationContext


async def test_rag_system():
    """
    Test the complete RAG system pipeline
    """
    print("🧪 Testing Movie Recommendation RAG System")
    print("=" * 50)
    
    # Step 1: Load movie data
    print("\n📊 Step 1: Loading movie data...")
    try:
        loader = MovieDataLoader(data_dir="../data")
        
        # Check if dataset exists, download if needed
        dataset_path = loader.data_dir / loader.dataset_name
        if not dataset_path.exists():
            print("📥 Downloading dataset...")
            if not loader.download_dataset():
                print("❌ Failed to download dataset")
                return False
        
        # Get sample data
        ratings_df, movies_df, user_movie_matrix = loader.get_sample_data(n_users=50, n_movies=20)
        print(f"✅ Loaded {len(movies_df)} movies and {len(ratings_df)} ratings")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Step 2: Initialize Knowledge Base
    print("\n🧠 Step 2: Initializing Knowledge Base...")
    try:
        kb = MovieKnowledgeBase(persist_directory="../data/test_vectorstore")
        
        # Build knowledge base (this may take a moment)
        print("🔄 Building knowledge base... (this may take a moment)")
        kb.build_knowledge_base(movies_df, force_rebuild=True)
        print("✅ Knowledge base created successfully")
        
    except Exception as e:
        print(f"❌ Error creating knowledge base: {e}")
        return False
    
    # Step 3: Initialize RAG Chain
    print("\n🤖 Step 3: Initializing RAG Chain...")
    try:
        rag = MovieRecommendationRAG(kb)
        print("✅ RAG chain initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {e}")
        return False
    
    # Step 4: Test Knowledge Base Search
    print("\n🔍 Step 4: Testing Knowledge Base Search...")
    try:
        # Test similarity search
        search_results = kb.search_similar_movies("action adventure movie", k=3)
        print(f"✅ Found {len(search_results)} similar movies")
        
        for i, result in enumerate(search_results):
            print(f"  {i+1}. {result['metadata']['title']} (Score: {result['similarity_score']:.3f})")
            
    except Exception as e:
        print(f"❌ Error in similarity search: {e}")
        return False
    
    # Step 5: Test RAG Explanation Generation
    print("\n💬 Step 5: Testing RAG Explanation Generation...")
    try:
        # Create a sample recommendation context
        sample_movie = movies_df.iloc[0]
        sample_user_id = 123
        
        # Create mock user preferences
        user_preferences = {
            "top_genres": {"Action": {"mean": 4.5, "count": 10}, "Adventure": {"mean": 4.2, "count": 8}},
            "average_rating": 4.1,
            "total_ratings": 25
        }
        
        # Get similar movies for context
        similar_movies = kb.search_similar_movies(f"movies like {sample_movie['title']}", k=3)
        
        context = RecommendationContext(
            user_id=sample_user_id,
            recommended_movie_id=int(sample_movie['movieId']),
            recommended_movie_title=sample_movie['title'],
            recommendation_score=4.2,
            recommendation_type="hybrid",
            user_preferences=user_preferences,
            similar_movies=similar_movies
        )
        
        # Generate explanation
        explanation = rag.generate_explanation(context)
        print(f"✅ Generated explanation successfully")
        print(f"📝 Sample explanation: {explanation}")
        
        # Get explanation metadata
        metadata = rag.get_explanation_metadata(context)
        print(f"🔧 Explanation metadata: {metadata}")
        
    except Exception as e:
        print(f"❌ Error generating explanation: {e}")
        return False
    
    # Step 6: Test Movie Context Retrieval
    print("\n📄 Step 6: Testing Movie Context Retrieval...")
    try:
        sample_movie_id = int(movies_df.iloc[0]['movieId'])
        context = kb.get_movie_context(sample_movie_id)
        
        if context:
            print("✅ Successfully retrieved movie context")
            print(f"📋 Context preview: {context[:200]}...")
        else:
            print("⚠️  No context found, but system handled gracefully")
            
    except Exception as e:
        print(f"❌ Error retrieving movie context: {e}")
        return False
    
    # Step 7: Performance Test
    print("\n⚡ Step 7: Performance Test...")
    try:
        import time
        
        # Test batch explanation generation
        start_time = time.time()
        
        contexts = []
        for i in range(3):  # Test with 3 movies
            movie = movies_df.iloc[i]
            context = RecommendationContext(
                user_id=123,
                recommended_movie_id=int(movie['movieId']),
                recommended_movie_title=movie['title'],
                recommendation_score=4.0 + i * 0.1,
                recommendation_type="hybrid",
                user_preferences=user_preferences,
                similar_movies=[]
            )
            contexts.append(context)
        
        explanations = rag.batch_generate_explanations(contexts)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Generated {len(explanations)} explanations in {processing_time:.2f} seconds")
        print(f"⚡ Average time per explanation: {processing_time/len(explanations):.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error in performance test: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! RAG system is working correctly!")
    print("\nSystem Components Status:")
    print(f"  📊 Movie Data: ✅ {len(movies_df)} movies loaded")
    print(f"  🧠 Knowledge Base: ✅ Vector store created")
    print(f"  🤖 RAG Chain: ✅ {'OpenAI' if rag.llm else 'Local fallback'} mode")
    print(f"  🔍 Search: ✅ Semantic search working")
    print(f"  💬 Explanations: ✅ Generation working")
    
    return True


async def main():
    """Main test function"""
    try:
        success = await test_rag_system()
        if success:
            print("\n✅ RAG system test completed successfully!")
            print("🚀 Your system is ready for Phase 2 deployment!")
        else:
            print("\n❌ RAG system test failed!")
            print("🔧 Please check the error messages above and fix any issues.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")


if __name__ == "__main__":
    # Set environment variables for testing (optional)
    os.environ.setdefault("OPENAI_API_KEY", "your_openai_api_key_here")
    
    print("🎬 Movie Recommendation RAG System Test")
    print("This will test all components of the enhanced recommendation system")
    print("Press Ctrl+C to cancel\n")
    
    # Run the test
    asyncio.run(main()) 