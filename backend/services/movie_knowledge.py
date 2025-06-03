"""
🎬 Movie Knowledge Base Service

This module creates and manages a comprehensive knowledge base of movie information
for our RAG (Retrieval-Augmented Generation) system. It stores detailed movie data
including plots, cast, reviews, and genre information that can be retrieved to provide
context for intelligent recommendation explanations.

Learning Objectives:
- Understand document preparation for RAG systems
- Learn about text chunking and embeddings
- Implement semantic search with vector databases
- Practice data preprocessing for LLM consumption
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

# For generating synthetic movie data (since we don't have full TMDB access)
import requests
from time import sleep


@dataclass
class MovieInfo:
    """
    Enhanced movie information structure for RAG system
    """
    movie_id: int
    title: str
    year: Optional[int]
    genres: List[str]
    plot: str
    cast: List[str]
    director: str
    rating: float
    popularity: float
    keywords: List[str]
    review_summary: str


class MovieKnowledgeBase:
    """
    Movie Knowledge Base for RAG System
    
    This class manages a comprehensive database of movie information that serves
    as the knowledge source for generating intelligent recommendation explanations.
    """
    
    def __init__(self, persist_directory: str = "data/movie_vectorstore"):
        """
        Initialize the Movie Knowledge Base
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = None
        
        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings (we'll use HuggingFace for local processing)
        self._initialize_embeddings()
        
        print("🏗️  Movie Knowledge Base initialized")
    
    def _initialize_embeddings(self):
        """Initialize embedding model for vector storage"""
        try:
            # Try to use OpenAI embeddings if API key is available
            if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
                print("🔑 Using OpenAI embeddings")
                self.embeddings = OpenAIEmbeddings()
            else:
                # Fallback to local HuggingFace embeddings
                print("🏠 Using local HuggingFace embeddings")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
        except Exception as e:
            print(f"⚠️  Error initializing embeddings: {e}")
            # Default to HuggingFace
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def create_synthetic_movie_data(self, movies_df: pd.DataFrame) -> List[MovieInfo]:
        """
        Create synthetic detailed movie information for demonstration
        
        In a real application, this would fetch data from TMDB, OMDB, or other APIs.
        For this demo, we'll create realistic movie information.
        
        Args:
            movies_df: DataFrame with basic movie information
            
        Returns:
            List of MovieInfo objects with enhanced data
        """
        print("📚 Creating enhanced movie knowledge base...")
        
        enhanced_movies = []
        
        # Sample plot templates for different genres
        plot_templates = {
            "Action": [
                "An action-packed thriller where our hero must overcome impossible odds to save the day.",
                "High-octane adventure featuring intense chase sequences and spectacular stunts.",
                "A gripping tale of survival and courage in the face of overwhelming danger."
            ],
            "Comedy": [
                "A hilarious misadventure that will keep you laughing from start to finish.",
                "A heartwarming comedy about friendship, love, and life's unexpected turns.",
                "A clever and witty story that offers both laughs and meaningful insights."
            ],
            "Drama": [
                "A powerful and emotional journey exploring the depths of human experience.",
                "An intimate character study that reveals the complexities of relationships.",
                "A thought-provoking narrative that challenges perspectives and touches hearts."
            ],
            "Horror": [
                "A chilling tale that will keep you on the edge of your seat.",
                "A supernatural thriller filled with suspense and unexpected scares.",
                "A psychological horror that explores the darkest corners of the mind."
            ],
            "Romance": [
                "A beautiful love story that celebrates the power of connection and hope.",
                "An enchanting romantic tale filled with passion and heartfelt moments.",
                "A touching story about finding love in the most unexpected places."
            ]
        }
        
        # Sample cast and director names
        sample_actors = [
            "Emma Stone", "Ryan Gosling", "Jennifer Lawrence", "Chris Evans",
            "Scarlett Johansson", "Leonardo DiCaprio", "Margot Robbie", "Tom Hanks",
            "Meryl Streep", "Brad Pitt", "Sandra Bullock", "Will Smith"
        ]
        
        sample_directors = [
            "Christopher Nolan", "Greta Gerwig", "Denis Villeneuve", "Jordan Peele",
            "Taika Waititi", "Rian Johnson", "Chloe Zhao", "Barry Jenkins"
        ]
        
        for _, movie in movies_df.head(100).iterrows():  # Process first 100 movies
            try:
                # Parse genres
                genres = [g.strip() for g in movie['genres'].split('|') if g.strip()]
                primary_genre = genres[0] if genres else "Drama"
                
                # Extract year from title
                year = None
                if '(' in movie['title'] and ')' in movie['title']:
                    try:
                        year_str = movie['title'].split('(')[-1].split(')')[0]
                        year = int(year_str) if year_str.isdigit() else None
                    except:
                        pass
                
                # Generate plot based on primary genre
                plot_options = plot_templates.get(primary_genre, plot_templates["Drama"])
                plot = f"{plot_options[movie['movieId'] % len(plot_options)]} Set in a {primary_genre.lower()} context, this film explores themes of {', '.join(genres[:2]).lower()}."
                
                # Generate cast and crew
                cast = [sample_actors[i % len(sample_actors)] for i in range(movie['movieId'], movie['movieId'] + 3)]
                director = sample_directors[movie['movieId'] % len(sample_directors)]
                
                # Generate keywords
                keywords = genres + [f"{primary_genre.lower()}_film", "entertainment", "cinema"]
                
                # Generate review summary
                review_summary = f"Critics praise this {primary_genre.lower()} for its engaging story and strong performances. A well-crafted film that resonates with audiences."
                
                # Create MovieInfo object
                movie_info = MovieInfo(
                    movie_id=movie['movieId'],
                    title=movie['title'],
                    year=year,
                    genres=genres,
                    plot=plot,
                    cast=cast,
                    director=director,
                    rating=3.5,  # Default rating since we don't have average_rating in this context
                    popularity=0,  # Default popularity
                    keywords=keywords,
                    review_summary=review_summary
                )
                
                enhanced_movies.append(movie_info)
                
            except Exception as e:
                print(f"⚠️  Error processing movie {movie.get('title', 'Unknown')}: {e}")
                continue
        
        print(f"✅ Created enhanced data for {len(enhanced_movies)} movies")
        return enhanced_movies
    
    def create_documents(self, movie_data: List[MovieInfo]) -> List[Document]:
        """
        Convert movie data into LangChain Documents for vector storage
        
        Args:
            movie_data: List of MovieInfo objects
            
        Returns:
            List of LangChain Document objects
        """
        print("📄 Creating documents for vector storage...")
        
        documents = []
        
        for movie in movie_data:
            # Create comprehensive text representation
            content = f"""
Title: {movie.title}
Year: {movie.year or 'Unknown'}
Genres: {', '.join(movie.genres)}
Director: {movie.director}
Cast: {', '.join(movie.cast)}
Rating: {movie.rating}/5.0

Plot: {movie.plot}

Review Summary: {movie.review_summary}

Keywords: {', '.join(movie.keywords)}
            """.strip()
            
            # Create metadata for filtering and retrieval
            # Note: Convert lists to strings for Chroma compatibility
            metadata = {
                "movie_id": movie.movie_id,
                "title": movie.title,
                "year": movie.year,
                "genres": ', '.join(movie.genres),  # Convert list to string
                "director": movie.director,
                "rating": movie.rating,
                "popularity": movie.popularity,
                "primary_genre": movie.genres[0] if movie.genres else "Unknown"
            }
            
            # Create Document object
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} documents")
        return documents
    
    def build_knowledge_base(self, movies_df: pd.DataFrame, force_rebuild: bool = False):
        """
        Build the vector store knowledge base
        
        Args:
            movies_df: DataFrame with movie information
            force_rebuild: Whether to rebuild even if existing vectorstore found
        """
        print("🏗️  Building movie knowledge base...")
        
        # Check if vectorstore already exists
        vectorstore_path = Path(self.persist_directory)
        if vectorstore_path.exists() and not force_rebuild:
            print("📂 Loading existing vector store...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                print("✅ Vector store loaded successfully")
                return
            except Exception as e:
                print(f"⚠️  Error loading existing vectorstore: {e}")
                print("🔄 Rebuilding vectorstore...")
        
        # Create enhanced movie data
        movie_data = self.create_synthetic_movie_data(movies_df)
        
        # Convert to documents
        documents = self.create_documents(movie_data)
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        split_documents = text_splitter.split_documents(documents)
        print(f"📝 Split into {len(split_documents)} chunks")
        
        # Filter complex metadata to ensure Chroma compatibility
        filtered_documents = filter_complex_metadata(split_documents)
        print(f"🔧 Filtered metadata for {len(filtered_documents)} documents")
        
        # Create vector store
        print("🔄 Creating vector store...")
        try:
            self.vectorstore = Chroma.from_documents(
                documents=filtered_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✅ Vector store created successfully")
            
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def search_similar_movies(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for movies similar to the query
        
        Args:
            query: Search query (can be genre, plot description, etc.)
            k: Number of results to return
            
        Returns:
            List of similar movie information
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Call build_knowledge_base first.")
        
        try:
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            similar_movies = []
            for doc, score in results:
                movie_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                }
                similar_movies.append(movie_info)
            
            return similar_movies
            
        except Exception as e:
            print(f"❌ Error in similarity search: {e}")
            return []
    
    def get_movie_context(self, movie_id: int) -> Optional[str]:
        """
        Get detailed context for a specific movie
        
        Args:
            movie_id: ID of the movie
            
        Returns:
            Detailed movie context string
        """
        if not self.vectorstore:
            return None
        
        try:
            # Search for the specific movie
            results = self.vectorstore.get(
                where={"movie_id": movie_id}
            )
            
            if results and results['documents']:
                return results['documents'][0]
            else:
                # Fallback: search by similarity
                results = self.vectorstore.similarity_search(
                    f"movie_id:{movie_id}",
                    k=1,
                    filter={"movie_id": movie_id}
                )
                return results[0].page_content if results else None
                
        except Exception as e:
            print(f"❌ Error getting movie context: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # This would typically be called from the main application
    print("🎬 Movie Knowledge Base Demo")
    
    # Initialize knowledge base
    kb = MovieKnowledgeBase()
    
    # In a real scenario, you'd load your movies DataFrame here
    # kb.build_knowledge_base(movies_df)
    
    print("✅ Knowledge base ready for RAG implementation!") 