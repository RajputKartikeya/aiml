"use client";

import React, { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { MovieCard } from "./MovieCard";
import {
  Movie,
  Recommendation,
  UserProfile,
  recommendationApi,
  movieApi,
  userApi,
  healthApi,
  genreUtils,
} from "@/lib/api";
import {
  Search,
  User,
  Film,
  Sparkles,
  AlertCircle,
  RefreshCw,
  TrendingUp,
  Heart,
} from "lucide-react";

export function RecommendationDashboard() {
  // State management
  const [currentUserId, setCurrentUserId] = useState<number>(219); // Default user from our backend
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [movies, setMovies] = useState<Movie[]>([]);
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<string>("checking...");

  // Search and filter states
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedGenre, setSelectedGenre] = useState<string>("all");
  const [recommendationType, setRecommendationType] = useState<
    "user" | "item" | "hybrid"
  >("hybrid");

  // Available genres for filtering
  const [availableGenres, setAvailableGenres] = useState<string[]>([]);

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
    loadInitialData();
  }, []);

  // Load user profile when user changes
  useEffect(() => {
    if (currentUserId) {
      loadUserProfile();
    }
  }, [currentUserId]);

  const checkApiHealth = async () => {
    try {
      const health = await healthApi.getHealth();
      setApiStatus(health.model_loaded ? "ready" : "model not loaded");
    } catch (error) {
      setApiStatus("offline");
      setError(
        "Cannot connect to the recommendation API. Make sure the backend is running!"
      );
    }
  };

  const loadInitialData = async () => {
    setLoading(true);
    try {
      // Load movies and extract genres
      const movieData = await movieApi.getMovies({ limit: 100 });
      setMovies(movieData);

      const genres = genreUtils.extractGenres(movieData);
      setAvailableGenres(genres);

      setError(null);
    } catch (error) {
      setError("Failed to load movie data");
      console.error("Error loading initial data:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadUserProfile = async () => {
    try {
      const profile = await userApi.getUserProfile(currentUserId);
      setUserProfile(profile);
    } catch (error) {
      console.error("Error loading user profile:", error);
      setUserProfile(null);
    }
  };

  const loadRecommendations = async () => {
    if (!currentUserId) return;

    setLoading(true);
    try {
      const response = await recommendationApi.getRecommendations({
        user_id: currentUserId,
        num_recommendations: 12,
        recommendation_type: recommendationType,
      });

      setRecommendations(response.recommendations);
      setError(null);
    } catch (error) {
      setError(`Failed to load recommendations: ${error}`);
      console.error("Error loading recommendations:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleRateMovie = async (movieId: number, rating: number) => {
    try {
      await movieApi.rateMovie({
        user_id: currentUserId,
        movie_id: movieId,
        rating: rating,
      });

      // Optionally reload recommendations after rating
      // loadRecommendations();
    } catch (error) {
      console.error("Error rating movie:", error);
      setError("Failed to submit rating");
    }
  };

  const searchMovies = async () => {
    setLoading(true);
    try {
      const params: any = { limit: 50 };
      if (searchTerm) params.search = searchTerm;
      if (selectedGenre && selectedGenre !== "all")
        params.genre = selectedGenre;

      const movieData = await movieApi.getMovies(params);
      setMovies(movieData);
      setError(null);
    } catch (error) {
      setError("Failed to search movies");
      console.error("Error searching movies:", error);
    } finally {
      setLoading(false);
    }
  };

  const renderUserProfile = () => {
    if (!userProfile) return null;

    const topGenres = Object.entries(userProfile.top_genres)
      .slice(0, 5)
      .map(([genre, stats]) => ({ genre, ...stats }));

    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="w-5 h-5" />
            User Profile
          </CardTitle>
          <CardDescription>Your movie preferences and activity</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {userProfile.total_ratings}
              </div>
              <div className="text-sm text-muted-foreground">Movies Rated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {userProfile.average_rating.toFixed(1)}
              </div>
              <div className="text-sm text-muted-foreground">Avg Rating</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary">
                {topGenres.length}
              </div>
              <div className="text-sm text-muted-foreground">Fav Genres</div>
            </div>
          </div>

          {topGenres.length > 0 && (
            <div className="mt-4">
              <div className="text-sm font-medium mb-2">Favorite Genres:</div>
              <div className="flex flex-wrap gap-2">
                {topGenres.map(({ genre, mean, count }) => (
                  <Badge key={genre} variant="secondary" className="text-xs">
                    {genre} ({mean.toFixed(1)}⭐, {count})
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
          🎬 AI Movie Recommendations
        </h1>
        <p className="text-lg text-muted-foreground">
          Discover your next favorite movie with AI-powered suggestions
        </p>

        {/* API Status */}
        <div className="mt-4">
          <Badge
            variant={apiStatus === "ready" ? "default" : "destructive"}
            className="text-xs"
          >
            API Status: {apiStatus}
          </Badge>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* User Selection */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>User Settings</CardTitle>
          <CardDescription>
            Select a user to get personalized recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 items-end">
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">User ID</label>
              <Input
                type="number"
                value={currentUserId}
                onChange={(e) => setCurrentUserId(Number(e.target.value))}
                placeholder="Enter user ID (e.g., 219)"
                min="1"
              />
            </div>
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">
                Recommendation Type
              </label>
              <Select
                value={recommendationType}
                onValueChange={(value: any) => setRecommendationType(value)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="hybrid">
                    🔀 Hybrid (Best Results)
                  </SelectItem>
                  <SelectItem value="user">👥 User-Based</SelectItem>
                  <SelectItem value="item">🎬 Item-Based</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <Button
              onClick={loadRecommendations}
              disabled={loading || !currentUserId || apiStatus !== "ready"}
              className="flex items-center gap-2"
            >
              {loading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              Get Recommendations
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* User Profile */}
      {renderUserProfile()}

      {/* Main Content Tabs */}
      <Tabs defaultValue="recommendations" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger
            value="recommendations"
            className="flex items-center gap-2"
          >
            <TrendingUp className="w-4 h-4" />
            Recommendations
          </TabsTrigger>
          <TabsTrigger value="browse" className="flex items-center gap-2">
            <Film className="w-4 h-4" />
            Browse Movies
          </TabsTrigger>
        </TabsList>

        {/* Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                Your Personalized Recommendations
              </CardTitle>
              <CardDescription>
                {recommendations.length > 0
                  ? `Here are ${recommendations.length} movies we think you'll love!`
                  : 'Click "Get Recommendations" to see personalized movie suggestions'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {recommendations.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {recommendations
                    .filter(
                      (movie, index, arr) =>
                        arr.findIndex((m) => m.movie_id === movie.movie_id) ===
                        index
                    )
                    .map((movie, index) => (
                      <MovieCard
                        key={`rec-${movie.movie_id}-${index}`}
                        movie={movie}
                        onRate={handleRateMovie}
                        isRecommendation={true}
                        showExplanation={true}
                      />
                    ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Heart className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                  <p className="text-lg text-muted-foreground mb-4">
                    No recommendations yet
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Enter a user ID and click "Get Recommendations" to see
                    personalized movie suggestions
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Browse Movies Tab */}
        <TabsContent value="browse" className="space-y-6">
          {/* Search and Filter */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="w-5 h-5" />
                Explore Movies
              </CardTitle>
              <CardDescription>
                Search and filter through our movie collection
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4 items-end">
                <div className="flex-1">
                  <label className="text-sm font-medium mb-2 block">
                    Search Movies
                  </label>
                  <Input
                    placeholder="Search movie titles..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    onKeyPress={(e) => e.key === "Enter" && searchMovies()}
                  />
                </div>
                <div className="flex-1">
                  <label className="text-sm font-medium mb-2 block">
                    Filter by Genre
                  </label>
                  <Select
                    value={selectedGenre}
                    onValueChange={setSelectedGenre}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="All genres" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Genres</SelectItem>
                      {availableGenres.map((genre) => (
                        <SelectItem key={genre} value={genre}>
                          {genre}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button onClick={searchMovies} disabled={loading}>
                  {loading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Search className="w-4 h-4" />
                  )}
                  Search
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Movies Grid */}
          <Card>
            <CardHeader>
              <CardTitle>Movie Collection</CardTitle>
              <CardDescription>
                {movies.length > 0
                  ? `Showing ${movies.length} movies`
                  : "No movies to display"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {movies.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                  {movies
                    .filter(
                      (movie, index, arr) =>
                        arr.findIndex((m) => m.movie_id === movie.movie_id) ===
                        index
                    )
                    .map((movie, index) => (
                      <MovieCard
                        key={`movie-${movie.movie_id}-${index}`}
                        movie={movie}
                        onRate={handleRateMovie}
                        isRecommendation={false}
                        showExplanation={false}
                      />
                    ))}
                </div>
              ) : (
                <div className="text-center py-12">
                  <Film className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
                  <p className="text-lg text-muted-foreground mb-4">
                    No movies found
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Try adjusting your search terms or filters
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default RecommendationDashboard;
