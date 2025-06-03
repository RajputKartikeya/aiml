#!/usr/bin/env python3
"""
🚀 Server Startup Script

Simple script to start our movie recommendation API server.
This makes it easy for beginners to get the server running!
"""

import uvicorn
import sys
import os

# Add current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_server():
    """Start the FastAPI server with proper configuration."""
    print("🎬 Starting Movie Recommendation API Server...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🔧 Alternative documentation at: http://localhost:8000/redoc")
    print("🌐 API endpoints at: http://localhost:8000/")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,  # Auto-reload on code changes during development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        print("💡 Make sure you've installed all requirements: pip install fastapi uvicorn")

if __name__ == "__main__":
    start_server() 