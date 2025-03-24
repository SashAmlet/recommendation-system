import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_recall_fscore_support

def get_data(path="data\\rawdata\\", N=10_000):
    """
    Load and preprocess movie ratings and metadata.
    This function reads movie ratings and metadata from CSV files, processes the data
    by removing unnecessary columns, handling missing values, and filtering the top N
    most popular movies based on the number of ratings.
    Args:
        path (str): The directory path where the CSV files are located. Defaults to "data\\rawdata\\".
        N (int): The number of most popular movies to retain. Defaults to 10,000.
    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - ratings (DataFrame): Processed ratings data with columns ['userId', 'movieId', 'rating'].
            - movies (DataFrame): Processed movies metadata with columns ['movieId', 'title', 'genres'].
    """
    # Load data
    ratings = pd.read_csv(f"{path}ratings.csv")  # userId, movieId, rating
    movies = pd.read_csv(f"{path}movies.csv")

    # Remove the timestamp column from ratings
    ratings = ratings.drop(columns=["timestamp"], errors="ignore")

    # Transform genres: split the genre string into a list
    movies["genres"] = movies["genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])

    # Check for missing values
    ratings.dropna(inplace=True)
    movies.dropna(inplace=True)

    # Calculate movie popularity based on the number of ratings
    movie_popularity = ratings["movieId"].value_counts().sort_values(ascending=False)

    # Keep only the N most popular movies
    top_movies = movie_popularity.head(N).index
    ratings = ratings[ratings["movieId"].isin(top_movies)]
    movies = movies[movies["movieId"].isin(top_movies)]

    return ratings, movies

def get_test_ds(ratings, output_ratings_file="data\\modified_ratings_10k.csv", output_indices_file="data\\removed_indices_10k.csv"):
    """
    Removes 20% of ratings from 20% of random users who have rated at least 12 movies.
    Saves the updated DataFrame and the pairs of removed userId-movieId to files.
    """
    # Group by userId and count the number of ratings for each user
    user_rating_counts = ratings.groupby("userId").size()

    # Select users who have rated at least 12 movies
    eligible_users = user_rating_counts[user_rating_counts >= 12].index

    # Select 20% of random users from the eligible ones
    selected_users = np.random.choice(eligible_users, size=int(len(eligible_users) * 0.2), replace=False)

    def process_user(user):
        user_ratings = ratings[ratings["userId"] == user]
        num_ratings_to_remove = int(len(user_ratings) * 0.2)

        # Select random indices for removal
        indices_to_remove = user_ratings.sample(num_ratings_to_remove, random_state=42).index
        return indices_to_remove

    # Parallelize the removal process across 4 cores
    removed_indices = Parallel(n_jobs=4)(delayed(process_user)(user) for user in selected_users)

    # Flatten the list of indices
    removed_indices = [index for sublist in removed_indices for index in sublist]

    # Create a new DataFrame without the removed ratings
    modified_ratings = ratings.drop(index=removed_indices)

    # Extract userId and movieId for the removed indices
    removed_pairs = ratings.loc[removed_indices, ["userId", "movieId"]]

    # Save the updated DataFrame and the pairs of removed userId-movieId
    modified_ratings.to_csv(output_ratings_file, index=False)
    removed_pairs.to_csv(output_indices_file, index=False)

    print(f"Modified ratings saved to {output_ratings_file}")
    print(f"Removed userId-movieId pairs saved to {output_indices_file}")


def evaluate_predictions(real_matrix, predicted_matrix, removed_indexes_file, threshold=3.5):
    """
    Считает RMSE, MAE, Precision, Recall и F1-score на основе реальных и предсказанных оценок.
    
    :param real_matrix: csr_matrix – реальная user-movie матрица
    :param predicted_matrix: csr_matrix – предсказанная user-movie матрица
    :param ratings_file: str – путь к файлу ratings.xlsx
    :param removed_indexes_file: str – путь к файлу removed_indexes.csv
    :param threshold: float – порог для положительных оценок
    :return: словарь с метриками
    """

    # Загружаем индексы предсказанных значений
    removed_indexes = pd.read_csv(removed_indexes_file)  # Changed from read_excel to read_csv

    # Извлекаем нужные userId и movieId (предсказанные оценки)
    removed_users = removed_indexes['userId'].values
    removed_movies = removed_indexes['movieId'].values

    # Фильтруем реальную и предсказанную матрицу
    y_true, y_pred = [], []

    for user, movie in zip(removed_users, removed_movies):
        real_rating = real_matrix[user, movie]
        predicted_rating = predicted_matrix[user, movie]

        if real_rating > 0:  # Если есть реальная оценка
            y_true.append(real_rating)
            y_pred.append(predicted_rating)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Считаем метрики
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # Для Precision / Recall считаем, какие фильмы "позитивные" (больше порога)
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }