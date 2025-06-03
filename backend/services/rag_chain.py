"""
🤖 RAG Chain for Intelligent Movie Recommendation Explanations

This module implements a Retrieval-Augmented Generation (RAG) chain that combines
our movie knowledge base with Large Language Model capabilities to generate
intelligent, contextual explanations for movie recommendations.

Learning Objectives:
- Understand RAG architecture and prompt engineering
- Learn to combine retrieval with generation
- Practice context management for LLMs
- Implement fallback strategies for robustness
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA
from langchain.llms.base import BaseLLM
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline

# Local imports
from .movie_knowledge import MovieKnowledgeBase


@dataclass
class RecommendationContext:
    """Context information for generating recommendations"""
    user_id: int
    recommended_movie_id: int
    recommended_movie_title: str
    recommendation_score: float
    recommendation_type: str
    user_preferences: Dict[str, Any]
    similar_movies: List[Dict[str, Any]]


class MovieRecommendationRAG:
    """
    RAG Chain for Movie Recommendation Explanations
    
    This class implements a sophisticated RAG system that retrieves relevant
    movie information and generates intelligent explanations for why specific
    movies are recommended to users.
    """
    
    def __init__(self, knowledge_base: MovieKnowledgeBase):
        """
        Initialize the RAG chain
        
        Args:
            knowledge_base: Initialized MovieKnowledgeBase instance
        """
        self.knowledge_base = knowledge_base
        self.llm = None
        self.retrieval_chain = None
        
        # Initialize LLM
        self._initialize_llm()
        
        # Create prompt templates
        self._create_prompts()
        
        print("🤖 Movie Recommendation RAG chain initialized")
    
    def _initialize_llm(self):
        """Initialize the Language Model"""
        try:
            # Try to use OpenAI if API key is available
            if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here":
                print("🔑 Using OpenAI GPT for explanations")
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.7,  # Some creativity but still focused
                    max_tokens=300    # Reasonable length for explanations
                )
            else:
                # Fallback to local/simplified explanations
                print("🏠 Using local explanation generation")
                self.llm = None  # We'll implement a simple template-based fallback
                
        except Exception as e:
            print(f"⚠️  Error initializing LLM: {e}")
            self.llm = None
    
    def _create_prompts(self):
        """Create prompt templates for different explanation scenarios"""
        
        # Main recommendation explanation prompt
        self.recommendation_prompt = PromptTemplate(
            input_variables=[
                "user_preferences", 
                "movie_info", 
                "recommendation_type", 
                "score",
                "similar_movies"
            ],
            template="""
You are a knowledgeable movie recommendation expert. Explain why a specific movie is being recommended to a user based on their preferences and viewing history.

User Preferences:
{user_preferences}

Recommended Movie Information:
{movie_info}

Recommendation Type: {recommendation_type}
Recommendation Score: {score}

Similar Movies the User Might Like:
{similar_movies}

Please provide a personalized, engaging explanation (2-3 sentences) for why this movie is recommended. Focus on:
1. How it matches their preferences
2. What makes it appealing
3. Connection to movies they've liked before

Explanation:"""
        )
        
        # Fallback template for when no LLM is available
        self.fallback_templates = {
            "user": "Based on users with similar tastes, this {genre} movie is highly rated by people who enjoyed {similar_genres}. The {recommendation_type} recommendation algorithm found strong similarities with your viewing preferences.",
            "item": "This movie shares key characteristics with films you've rated highly. It features {shared_elements} and has received positive reviews for its {strengths}.",
            "hybrid": "Our AI combines multiple signals to recommend this film. It matches your preference for {genres} and is similar to movies you've enjoyed like {similar_titles}."
        }
    
    def retrieve_movie_context(self, movie_id: int, user_context: str = "") -> str:
        """
        Retrieve relevant context about a movie for explanation generation
        
        Args:
            movie_id: ID of the recommended movie
            user_context: Additional context about user preferences
            
        Returns:
            Retrieved movie context string
        """
        try:
            # Get specific movie information
            movie_context = self.knowledge_base.get_movie_context(movie_id)
            
            if movie_context:
                return movie_context
            else:
                # Fallback: search for similar content
                query = f"movie_id:{movie_id} {user_context}"
                similar_results = self.knowledge_base.search_similar_movies(query, k=1)
                
                if similar_results:
                    return similar_results[0]["content"]
                
                return "Movie information not available in knowledge base."
                
        except Exception as e:
            print(f"❌ Error retrieving movie context: {e}")
            return "Unable to retrieve movie information."
    
    def generate_explanation(self, context: RecommendationContext) -> str:
        """
        Generate an intelligent explanation for a movie recommendation
        
        Args:
            context: RecommendationContext containing all relevant information
            
        Returns:
            Generated explanation string
        """
        try:
            # Retrieve movie context
            movie_context = self.retrieve_movie_context(
                context.recommended_movie_id,
                f"user preferences: {context.user_preferences}"
            )
            
            # If we have an LLM, use it for sophisticated explanations
            if self.llm:
                return self._generate_llm_explanation(context, movie_context)
            else:
                return self._generate_fallback_explanation(context, movie_context)
                
        except Exception as e:
            print(f"❌ Error generating explanation: {e}")
            return self._generate_simple_fallback(context)
    
    def _generate_llm_explanation(self, context: RecommendationContext, movie_context: str) -> str:
        """Generate explanation using LLM"""
        try:
            # Prepare context for the prompt
            user_prefs_text = self._format_user_preferences(context.user_preferences)
            similar_movies_text = self._format_similar_movies(context.similar_movies)
            
            # Format the prompt
            prompt_input = {
                "user_preferences": user_prefs_text,
                "movie_info": movie_context,
                "recommendation_type": context.recommendation_type,
                "score": f"{context.recommendation_score:.2f}",
                "similar_movies": similar_movies_text
            }
            
            # Generate response
            prompt = self.recommendation_prompt.format(**prompt_input)
            response = self.llm.invoke(prompt)
            
            # Extract text from response (handling different response types)
            if hasattr(response, 'content'):
                explanation = response.content
            else:
                explanation = str(response)
            
            # Clean up the explanation
            explanation = explanation.strip()
            if explanation.startswith("Explanation:"):
                explanation = explanation.replace("Explanation:", "").strip()
            
            return explanation
            
        except Exception as e:
            print(f"⚠️  Error with LLM generation, falling back: {e}")
            return self._generate_fallback_explanation(context, movie_context)
    
    def _generate_fallback_explanation(self, context: RecommendationContext, movie_context: str) -> str:
        """Generate explanation using template-based fallback"""
        try:
            # Extract key information from movie context
            genres = self._extract_genres_from_context(movie_context)
            
            # Get appropriate template
            template = self.fallback_templates.get(
                context.recommendation_type.split()[0], 
                self.fallback_templates["hybrid"]
            )
            
            # Fill in template variables
            explanation = template.format(
                genre=genres[0] if genres else "movie",
                genres=", ".join(genres[:2]),
                similar_genres=", ".join(genres[:2]),
                recommendation_type=context.recommendation_type,
                shared_elements="compelling storytelling and strong performances",
                strengths="engaging plot and character development",
                similar_titles="films in your viewing history"
            )
            
            return explanation
            
        except Exception as e:
            print(f"⚠️  Error with fallback generation: {e}")
            return self._generate_simple_fallback(context)
    
    def _generate_simple_fallback(self, context: RecommendationContext) -> str:
        """Generate a simple fallback explanation"""
        return f"This movie is recommended based on your viewing preferences and has a compatibility score of {context.recommendation_score:.2f}. Users with similar tastes have rated it highly."
    
    def _format_user_preferences(self, preferences: Dict[str, Any]) -> str:
        """Format user preferences for prompt"""
        try:
            pref_lines = []
            
            if "top_genres" in preferences:
                top_genres = list(preferences["top_genres"].keys())[:3]
                pref_lines.append(f"Favorite genres: {', '.join(top_genres)}")
            
            if "average_rating" in preferences:
                pref_lines.append(f"Average rating given: {preferences['average_rating']:.1f}/5.0")
            
            if "total_ratings" in preferences:
                pref_lines.append(f"Movies rated: {preferences['total_ratings']}")
            
            return "\n".join(pref_lines) if pref_lines else "Limited preference data available"
            
        except Exception:
            return "User preference information not available"
    
    def _format_similar_movies(self, similar_movies: List[Dict[str, Any]]) -> str:
        """Format similar movies information for prompt"""
        try:
            if not similar_movies:
                return "No similar movies data available"
            
            movie_lines = []
            for movie in similar_movies[:3]:  # Top 3 similar movies
                if "metadata" in movie:
                    metadata = movie["metadata"]
                    title = metadata.get("title", "Unknown")
                    genres = metadata.get("genres", "")
                    # Handle both string and list formats for genres
                    if isinstance(genres, str):
                        genres_display = genres.split(', ')[:2]  # Take first 2 genres
                        genres_str = ', '.join(genres_display)
                    else:
                        genres_str = ', '.join(genres[:2]) if genres else ""
                    movie_lines.append(f"- {title} ({genres_str})")
            
            return "\n".join(movie_lines) if movie_lines else "Similar movies data not available"
            
        except Exception:
            return "Similar movies information not available"
    
    def _extract_genres_from_context(self, context: str) -> List[str]:
        """Extract genres from movie context"""
        try:
            lines = context.split('\n')
            for line in lines:
                if line.startswith('Genres:'):
                    genres_text = line.replace('Genres:', '').strip()
                    # Handle both comma-separated and pipe-separated genres
                    if ', ' in genres_text:
                        return [g.strip() for g in genres_text.split(',') if g.strip()]
                    elif '|' in genres_text:
                        return [g.strip() for g in genres_text.split('|') if g.strip()]
                    else:
                        return [genres_text] if genres_text else ["Drama"]
            return ["Drama"]  # Default fallback
        except Exception:
            return ["Drama"]
    
    def batch_generate_explanations(self, contexts: List[RecommendationContext]) -> List[str]:
        """
        Generate explanations for multiple recommendations
        
        Args:
            contexts: List of RecommendationContext objects
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        for context in contexts:
            explanation = self.generate_explanation(context)
            explanations.append(explanation)
        
        return explanations
    
    def get_explanation_metadata(self, context: RecommendationContext) -> Dict[str, Any]:
        """
        Get metadata about the explanation generation process
        
        Args:
            context: RecommendationContext
            
        Returns:
            Dictionary with explanation metadata
        """
        return {
            "llm_available": self.llm is not None,
            "llm_type": "OpenAI GPT-3.5" if self.llm else "Template-based",
            "recommendation_type": context.recommendation_type,
            "confidence_score": context.recommendation_score,
            "context_retrieved": True,  # Could be dynamic based on retrieval success
            "explanation_length": "2-3 sentences",
            "personalization_level": "High" if self.llm else "Medium"
        }


# Example usage and testing
if __name__ == "__main__":
    print("🤖 RAG Chain Demo")
    
    # This would typically be initialized with a real knowledge base
    # from .movie_knowledge import MovieKnowledgeBase
    # kb = MovieKnowledgeBase()
    # rag = MovieRecommendationRAG(kb)
    
    print("✅ RAG chain ready for generating intelligent explanations!") 