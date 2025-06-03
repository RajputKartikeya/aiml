"use client";

import React, { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Movie, Recommendation, utils, genreUtils } from "@/lib/api";
import { Star, Info, ThumbsUp } from "lucide-react";

interface MovieCardProps {
  movie: Movie | Recommendation;
  onRate?: (movieId: number, rating: number) => void;
  isRecommendation?: boolean;
  showExplanation?: boolean;
}

export function MovieCard({
  movie,
  onRate,
  isRecommendation = false,
  showExplanation = false,
}: MovieCardProps) {
  const [userRating, setUserRating] = useState<number>(0);
  const [isRating, setIsRating] = useState(false);

  const cleanTitle = utils.cleanTitle(movie.title);
  const year = utils.extractYear(movie.title);
  const genres = genreUtils.formatGenres(movie.genres);

  const handleRateMovie = async (rating: number) => {
    if (onRate) {
      setIsRating(true);
      try {
        await onRate(movie.movie_id, rating);
        setUserRating(rating);
      } catch (error) {
        console.error("Failed to rate movie:", error);
      } finally {
        setIsRating(false);
      }
    }
  };

  const renderStarRating = () => {
    return (
      <div className="flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            onClick={() => handleRateMovie(star)}
            disabled={isRating}
            className={`w-5 h-5 transition-colors ${
              star <= userRating
                ? "text-yellow-400 hover:text-yellow-500"
                : "text-gray-300 hover:text-yellow-300"
            } disabled:opacity-50`}
          >
            <Star className="w-full h-full fill-current" />
          </button>
        ))}
      </div>
    );
  };

  const recommendationScore =
    isRecommendation && "score" in movie ? movie.score : null;
  const explanation =
    isRecommendation && "explanation" in movie ? movie.explanation : null;

  return (
    <Card className="w-full h-full flex flex-col transition-all duration-200 hover:shadow-lg hover:scale-105">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <CardTitle
              className="text-lg font-bold leading-tight line-clamp-2 break-words"
              title={cleanTitle}
            >
              {cleanTitle}
            </CardTitle>
            {year && (
              <CardDescription className="text-sm text-muted-foreground mt-1">
                {year}
              </CardDescription>
            )}
          </div>
          {isRecommendation && recommendationScore && (
            <div className="ml-2 text-right flex-shrink-0">
              <div className="text-sm font-semibold text-primary">
                {typeof recommendationScore === "number" &&
                recommendationScore < 10
                  ? `${utils.formatRating(recommendationScore)}${
                      recommendationScore < 5 ? "/5" : "%"
                    }`
                  : utils.formatRating(recommendationScore)}
              </div>
              <div className="text-xs text-muted-foreground">
                {recommendationScore < 5 ? "predicted" : "match"}
              </div>
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent className="pt-0 flex-1 flex flex-col justify-between">
        <div className="space-y-3">
          {/* Genres */}
          <div className="flex flex-wrap gap-1">
            {genres.slice(0, 2).map((genre) => (
              <Badge
                key={genre}
                variant="secondary"
                className="text-xs truncate max-w-20"
              >
                {genre}
              </Badge>
            ))}
            {genres.length > 2 && (
              <Badge variant="outline" className="text-xs">
                +{genres.length - 2}
              </Badge>
            )}
          </div>

          {/* Average Rating */}
          {"average_rating" in movie && movie.average_rating && (
            <div className="flex items-center gap-2">
              <div className="flex items-center">
                <Star className="w-4 h-4 text-yellow-400 fill-current" />
                <span className="ml-1 text-sm font-medium">
                  {utils.formatRating(movie.average_rating)}
                </span>
              </div>
              {"total_ratings" in movie && movie.total_ratings && (
                <span className="text-xs text-muted-foreground truncate">
                  ({movie.total_ratings} ratings)
                </span>
              )}
            </div>
          )}

          {/* User Rating */}
          {onRate && (
            <div>
              <div className="text-sm font-medium mb-1">Rate this movie:</div>
              {renderStarRating()}
            </div>
          )}
        </div>

        {/* Bottom Actions */}
        <div className="space-y-2 mt-4">
          {/* Recommendation Explanation */}
          {showExplanation && explanation && (
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm" className="w-full">
                  <Info className="w-4 h-4 mr-2" />
                  Why recommended?
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-md">
                <DialogHeader>
                  <DialogTitle>Why we recommend this movie</DialogTitle>
                  <DialogDescription className="text-left">
                    {explanation}
                  </DialogDescription>
                </DialogHeader>
              </DialogContent>
            </Dialog>
          )}

          {/* Recommendation Type Badge */}
          {isRecommendation && "recommendation_type" in movie && (
            <div className="flex justify-between items-center gap-2">
              <Badge variant="outline" className="text-xs truncate flex-1">
                {movie.recommendation_type
                  .replace("hybrid (", "")
                  .replace(")", "")}
              </Badge>
              {onRate && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() =>
                    handleRateMovie(userRating > 0 ? userRating : 4)
                  }
                  disabled={isRating}
                  className="flex-shrink-0"
                >
                  <ThumbsUp className="w-4 h-4 mr-1" />
                  Like
                </Button>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default MovieCard;
