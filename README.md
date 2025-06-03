# 🎬 AI Movie Recommendation System

A full-stack movie recommendation system built with Python/FastAPI backend and React/Next.js frontend, demonstrating machine learning, data science, and modern web development skills.

## 🎯 Project Overview

This project showcases skills specifically required for Data Science/ML Engineer roles, including:

- **Python expertise** with FastAPI for high-performance APIs
- **Machine Learning** with collaborative filtering using scikit-learn
- **Data manipulation** with Pandas and NumPy
- **Modern AI integration** ready for LangChain/RAG implementation
- **Full-stack development** with React, TypeScript, and beautiful UI components

## 🚀 Features Implemented

### ✅ Backend (Python/FastAPI)

- **Machine Learning Engine**: Collaborative filtering with user-based and item-based recommendations
- **Hybrid Recommendation System**: Combines multiple approaches for better accuracy
- **Data Processing Pipeline**: MovieLens dataset integration with comprehensive preprocessing
- **RESTful API**: Complete CRUD operations with automatic documentation
- **Model Evaluation**: RMSE/MAE metrics with train/test validation
- **Real-time Recommendations**: Instant personalized movie suggestions

### ✅ Frontend (React/Next.js)

- **Interactive Dashboard**: Beautiful, responsive movie recommendation interface
- **User Profiles**: Display user statistics, preferences, and rating history
- **Movie Rating System**: Interactive star ratings with hover effects
- **Search & Filter**: Advanced movie browsing with genre filtering
- **Recommendation Explanations**: AI-powered reasoning for why movies are suggested
- **Type-safe API Integration**: Full TypeScript support with proper error handling

### ✅ Technical Achievements

- **Model Performance**: RMSE of 0.886 on test data (excellent for movie ratings)
- **Data Scale**: 100,836+ ratings from 610+ users across 9,724+ movies
- **API Performance**: Fast, efficient recommendations with proper caching
- **UI/UX Excellence**: Modern, accessible interface with shadcn/ui components

## 🛠 Tech Stack

### Backend

- **Python 3.8+** - Core language
- **FastAPI** - High-performance API framework
- **scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation and analysis
- **Uvicorn** - ASGI server for production deployment

### Frontend

- **Next.js 15.3.3** - React framework with TypeScript
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Beautiful, accessible UI components
- **Lucide React** - Modern icon library

### Data & ML

- **MovieLens Dataset** - Real movie ratings data
- **Collaborative Filtering** - User-based and item-based algorithms
- **Cosine Similarity** - User preference matching
- **Matrix Factorization** - Advanced recommendation techniques

## 🏗 Project Structure

```
├── backend/                    # Python/FastAPI backend
│   ├── models/                # ML models and algorithms
│   │   └── recommender.py     # Collaborative filtering implementation
│   ├── services/              # Business logic
│   ├── api/                   # API endpoints and routes
│   ├── data/                  # Data processing utilities
│   │   └── data_loader.py     # MovieLens data handling
│   ├── config/                # Configuration management
│   ├── main.py                # FastAPI application entry point
│   ├── train_model.py         # ML training pipeline
│   └── start_server.py        # Development server launcher
├── frontend/                  # React/Next.js frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   │   ├── MovieCard.tsx  # Movie display component
│   │   │   └── RecommendationDashboard.tsx
│   │   ├── lib/               # API integration and utilities
│   │   │   └── api.ts         # Type-safe API client
│   │   └── app/               # Next.js app directory
│   ├── package.json           # Frontend dependencies
│   └── tailwind.config.js     # Styling configuration
├── data/                      # MovieLens dataset storage
├── models/                    # Trained model artifacts
├── docs/                      # Project documentation
├── tests/                     # Test suites
├── requirements.txt           # Python dependencies
├── env.example               # Environment variable template
└── README.md                 # This file
```

## 🚦 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
cd backend
python start_server.py
```

The backend will be available at: **http://localhost:8000**

- API documentation: **http://localhost:8000/docs**
- Alternative docs: **http://localhost:8000/redoc**

### 2. Frontend Setup

```bash
# Install Node.js dependencies
cd frontend
npm install

# Start the development server
npm run dev
```

The frontend will be available at: **http://localhost:3000**

### 3. Environment Configuration (Optional)

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your settings (API keys, etc.)
```

## 🎮 Usage

### Getting Recommendations

1. **Open http://localhost:3000** in your browser
2. **Enter a User ID** (try 219 for sample data)
3. **Select recommendation type**: Hybrid (recommended), User-based, or Item-based
4. **Click "Get Recommendations"** to see personalized movie suggestions
5. **Rate movies** by clicking the star ratings
6. **View explanations** by clicking "Why recommended?" buttons

### Exploring Movies

1. **Switch to "Browse Movies" tab**
2. **Search by title** or **filter by genre**
3. **Rate movies** to improve future recommendations
4. **Discover new favorites** through the curated collection

## 📊 Machine Learning Details

### Algorithms Implemented

- **User-Based Collaborative Filtering**: Finds similar users and recommends their favorite movies
- **Item-Based Collaborative Filtering**: Recommends movies similar to ones you've already rated highly
- **Hybrid Approach**: Combines both methods for optimal results

### Model Performance

- **RMSE**: 0.886 (excellent for movie rating predictions)
- **Dataset**: 100,836+ ratings across 9,724+ movies
- **Cold Start Handling**: Graceful degradation for new users
- **Scalability**: Efficient algorithms suitable for production use

### Data Processing

- **Automatic Downloads**: MovieLens dataset fetched automatically
- **Data Cleaning**: Handles missing values and outliers
- **Feature Engineering**: Creates user-movie interaction matrices
- **Evaluation**: Proper train/test splits with cross-validation

## 🔮 Roadmap (Phase 2)

### Planned Features

- **LangChain Integration**: RAG-powered movie explanations
- **Vector Database**: Semantic search for movie content
- **Advanced ML**: Deep learning models with TensorFlow/PyTorch
- **Real-time Updates**: Live recommendation updates
- **A/B Testing**: Compare different recommendation strategies

## 🤝 Contributing

This project is designed for learning and demonstration. Feel free to:

- Fork the repository
- Submit pull requests
- Report issues
- Suggest improvements

## 📝 Technical Notes

### API Endpoints

- `GET /health` - Check API status
- `GET /movies` - List movies with filtering
- `POST /recommendations` - Get personalized recommendations
- `POST /rate` - Submit movie ratings
- `GET /users/{user_id}/profile` - Get user profile

### Key Features

- **Type Safety**: Full TypeScript support throughout
- **Error Handling**: Comprehensive error management
- **Performance**: Optimized for speed and scalability
- **Accessibility**: WCAG-compliant UI components
- **Responsive Design**: Works on all device sizes

## 🎓 Learning Objectives

This project demonstrates:

- **Python Proficiency**: Advanced Python programming with ML libraries
- **Data Science Skills**: Real-world data processing and analysis
- **Machine Learning**: Implementation of recommendation algorithms
- **API Development**: RESTful services with proper documentation
- **Frontend Development**: Modern React with TypeScript
- **Full-Stack Integration**: Seamless frontend-backend communication
- **UI/UX Design**: Beautiful, intuitive user interfaces

Perfect for showcasing skills required for Data Science, ML Engineer, and Full-Stack Developer roles!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
