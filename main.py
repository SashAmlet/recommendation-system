import pandas as pd
import numpy as np
from data import get_data, get_test_ds, evaluate_predictions
from visualization import MovieDataVisualizer
from recommender import MovieRecommender

real_ratings, movies = get_data(N=10_000)

testing_ratings =  get_test_ds(real_ratings, generate_new=False)

# visualizer = MovieDataVisualizer(real_ratings, movies)
# visualizer.plot_top_rated_movies()
# visualizer.plot_movie_ratings()
# visualizer.plot_rating_distribution()
# visualizer.plot_top_users()
# visualizer.plot_user_types()

recommendation = MovieRecommender(real_ratings, testing_ratings)
recommendation.train(method = "MatrixFactorization", generate_new=False)

result = evaluate_predictions(recommendation.real_user_movie, recommendation.predicted_user_movie, "data\\removed_indices.csv")

print(result)