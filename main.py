import pandas as pd
import numpy as np
from data import get_data, get_test_ds, evaluate_predictions
from visualization import MovieDataVisualizer
from recommender import MovieRecommender

ratings, movies = get_data()
testing_ratings = pd.read_csv("data\\modified_ratings_10k.csv")

# get_test_ds(ratings, output_ratings_file="data\\modified_ratings_10k.csv", output_indices_file="data\\removed_indices_10k.csv")

# print(ratings.head())
# print(movies.head())

# visualizer = MovieDataVisualizer(ratings, movies)
# visualizer.plot_top_rated_movies()
# visualizer.plot_movie_ratings()
# visualizer.plot_rating_distribution()
# visualizer.plot_top_users()
# visualizer.plot_user_types()

recommendation = MovieRecommender(ratings, testing_ratings)
recommendation.train(generate_new=False, filename="data\\data_predicted_ratings_10k.npz")

result = evaluate_predictions(recommendation.real_user_movie, recommendation.predicted_user_movie, "data\\removed_indices_10k.csv")

print(result)