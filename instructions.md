# Cursor AI Assistant: Build a Movie Recommendation System with RAG Integration

## Nagarro looking for these skills

Required Skills:

- Demonstrate proficiency in programming languages such as Python or R, especially for data manipulation tasks using libraries like Pandas and NumPy.
- Apply expertise in machine learning algorithms and frameworks such as scikit-learn, TensorFlow, and PyTorch in practical scenarios.
- Utilize mathematical optimization techniques including linear programming, integer programming, genetic algorithms, and ant colony optimisation.
- Implement Reinforcement Learning models and understand their application in optimization and decision-making problems.
- Integrate machine learning models with large language models (LLMs) and work on generative AI applications.
- Design and develop Retrieval-Augmented Generation (RAG) pipelines using frameworks such as LangChain and LlamaIndex.
- Evaluate the performance of RAG systems and apply best practices in prompt engineering.
- Communicate technical concepts clearly and effectively to team members, stakeholders, or learners.
  Must have Skills : Python (Expert) Good To Have Skills : Machine Learning Fundamentals

## Project Overview

Create a full-stack movie recommendation system that demonstrates proficiency in machine learning, data manipulation, and modern AI integration. This project should showcase skills specifically required for a Data Science/ML Engineer role at Nagarro.

## Core Technical Requirements

### Data Science & Machine Learning Foundation

- **Primary Task**: Implement collaborative filtering using scikit-learn for movie recommendations
- **Data Manipulation**: Use Pandas and NumPy extensively for data preprocessing, analysis, and feature engineering
- **Mathematical Concepts**: Apply cosine similarity, matrix factorization, and basic statistical analysis
- **Performance Evaluation**: Implement metrics like RMSE, precision@k, and recall@k to evaluate recommendation quality
- **Explain every mathematical concept clearly**: When implementing algorithms, provide detailed comments explaining why each step works and what the mathematical operations accomplish

### Large Language Model Integration (Critical for Nagarro)

- **RAG Pipeline**: Build a Retrieval-Augmented Generation system using LangChain framework
- **Knowledge Base**: Create a structured database of movie information (plots, reviews, genre details, cast information)
- **Prompt Engineering**: Design effective prompts that generate personalized recommendation explanations
- **LLM Integration**: Connect with OpenAI API or use local models through Hugging Face transformers
- **Context Management**: Implement proper context window management for long conversations about recommendations

### Backend Architecture & APIs

- **Framework**: Use FastAPI for high-performance API development with automatic documentation
- **Database Design**: Implement both relational (PostgreSQL/MySQL) and vector database (for embeddings) integration
- **RESTful Services**: Create endpoints for user management, rating submission, recommendation generation, and explanation retrieval
- **Authentication**: Implement JWT-based authentication system for user sessions
- **Error Handling**: Build robust error handling and logging throughout the system

### Frontend Development

- **Framework**: Use React.js with TypeScript for type safety and better development experience
- **State Management**: Implement proper state management for user interactions and recommendation display
- **Data Visualization**: Create interactive charts showing recommendation confidence scores and user preference patterns
- **Responsive Design**: Build mobile-friendly interface using modern CSS frameworks
- **User Experience**: Design intuitive interfaces for rating movies and exploring recommendations with explanations

## Specific Implementation Requirements

### Data Processing Pipeline

```python
# Please implement these core data processing functions with detailed explanations:
# 1. Load and clean MovieLens dataset using Pandas
# 2. Create user-item interaction matrices using NumPy
# 3. Handle missing data and outliers appropriately
# 4. Generate embeddings for movie content using TensorFlow/PyTorch
# 5. Implement data validation and quality checks
```

### Machine Learning Models

```python
# Core ML implementations needed:
# 1. User-based collaborative filtering with cosine similarity
# 2. Item-based collaborative filtering with Pearson correlation
# 3. Matrix factorization using scikit-learn's NMF or SVD
# 4. Hybrid approach combining content-based and collaborative filtering
# 5. Model evaluation with cross-validation techniques
```

### RAG System Architecture

```python
# RAG pipeline components to implement:
# 1. Document chunking and preprocessing for movie information
# 2. Vector embeddings generation using sentence-transformers
# 3. Similarity search implementation for relevant context retrieval
# 4. Prompt template design for recommendation explanations
# 5. Response generation with proper citation of sources
```

## Learning Objectives Integration

### Demonstrate Understanding Through Code Comments

- Explain why specific algorithms were chosen for different recommendation scenarios
- Document the mathematical intuition behind similarity calculations and matrix operations
- Describe how RAG improves recommendation explainability compared to traditional methods
- Comment on optimization decisions and their impact on system performance

### Business Context Awareness

- Include comments explaining how each feature addresses real-world business problems
- Demonstrate understanding of recommendation system challenges like cold start problems and scalability
- Show awareness of ethical considerations in recommendation systems (filter bubbles, bias)
- Connect technical implementation to user experience improvements

## Advanced Features to Implement

### Optimization Techniques

- Implement basic genetic algorithm for hyperparameter tuning of recommendation models
- Use mathematical optimization for balancing different recommendation factors (popularity vs personalization)
- Apply reinforcement learning concepts for dynamic recommendation adjustment based on user feedback
- Implement caching strategies for improved system performance

### Model Integration Patterns

- Create ensemble methods combining multiple recommendation approaches
- Implement A/B testing framework for comparing different recommendation strategies
- Build monitoring dashboard for tracking recommendation system performance
- Design feedback loops for continuous model improvement

## Project Structure Guidelines

### File Organization

```
recommendation-system/
├── backend/
│   ├── models/          # ML models and training scripts
│   ├── services/        # Business logic and RAG implementation
│   ├── api/            # FastAPI endpoints
│   ├── data/           # Data processing utilities
│   └── config/         # Configuration and environment setup
├── frontend/
│   ├── components/     # React components
│   ├── services/       # API integration
│   ├── hooks/          # Custom React hooks
│   └── utils/          # Utility functions
├── notebooks/          # Jupyter notebooks for data exploration
├── tests/             # Comprehensive test suite
└── docs/              # Documentation and setup guides
```

### Code Quality Standards

- Include comprehensive docstrings for all functions explaining parameters, return values, and business logic
- Implement proper error handling with meaningful error messages
- Write unit tests for critical functions, especially ML model components
- Use type hints throughout Python code for better maintainability
- Follow PEP 8 style guidelines and include linting configuration

## Specific Nagarro Requirements Integration

### Technical Skills Demonstration

- **Python Expertise**: Showcase advanced Python features like decorators, context managers, and async programming
- **Data Manipulation**: Demonstrate complex Pandas operations including groupby, pivot tables, and time series analysis
- **ML Frameworks**: Use both scikit-learn for traditional ML and TensorFlow/PyTorch for deep learning components
- **LangChain Proficiency**: Implement multiple LangChain components including document loaders, retrievers, and chains

### Communication and Documentation

- Write clear README with setup instructions and architecture explanations
- Create API documentation using FastAPI's automatic documentation features
- Include inline comments explaining complex algorithms and business logic
- Prepare presentation materials showing system architecture and key features

## Implementation Priority Order

### Phase 1 (Days 1-2): Foundation

1. Set up project structure with proper virtual environment and dependencies
2. Load and explore MovieLens dataset with comprehensive data analysis
3. Implement basic collaborative filtering with detailed mathematical explanations
4. Create simple API endpoints for testing recommendation functionality

### Phase 2 (Days 3-4): AI Integration

1. Integrate LangChain for RAG pipeline implementation
2. Build vector database for movie information storage and retrieval
3. Implement prompt engineering for generating recommendation explanations
4. Test and refine RAG system for coherent and relevant explanations

### Phase 3 (Day 5): Polish and Integration

1. Complete frontend development with interactive recommendation interface
2. Implement comprehensive error handling and input validation
3. Add performance monitoring and basic analytics dashboard
4. Prepare documentation and presentation materials for interview discussion

## Success Metrics

- Recommendation accuracy measured by RMSE and precision metrics
- RAG system coherence evaluated through explanation quality
- System performance measured by response times and scalability
- Code quality assessed through documentation completeness and test coverage
- Business value demonstration through user experience improvements

Please implement this project step-by-step, providing detailed explanations for each component and ensuring that every technical decision aligns with demonstrating the specific skills Nagarro requires. Focus on creating a learning experience that builds understanding progressively while delivering a functional, impressive final product.
