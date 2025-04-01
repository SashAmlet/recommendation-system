import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz, diags
from scipy.sparse.linalg import svds 
from datetime import datetime
from implicit.als import AlternatingLeastSquares


class MovieRecommender:
    def __init__(self, real_ratings, modified_ratings):
        self.predicted_user_movie = None
        self.real_user_movie = csr_matrix(
                (real_ratings['rating'], (real_ratings['userId'], real_ratings['movieId']))
            )
        
        self.zeroed_user_movie = csr_matrix(
                (modified_ratings['rating'], (modified_ratings['userId'], modified_ratings['movieId']))
            )
        

    
    def train(self, method="SVD", generate_new=False, filename="data\\predicted_ratings_SVD.npz"):
        if not generate_new:
            self.predicted_user_movie = load_npz(filename)
        else:
            if method == "SVD":
                self.predicted_user_movie = self.predict_ratings_by_SVD()
            elif method == "ALS":
                self.predicted_user_movie = self.predict_ratings_by_ALS()
            else:
                raise ValueError("Unsupported method.")

            # Generate filename with method and date
            date_str = datetime.now().strftime("%Y%m%d")
            save_filename = f"data\\predicted_ratings_{method}_{date_str}.npz"
            save_npz(save_filename, self.predicted_user_movie)
        
    def predict_ratings_by_SVD(self, k=80) -> csr_matrix:
        """
        Predict user-movie ratings using Singular Value Decomposition (SVD).
        This method performs matrix factorization on the user-movie rating matrix 
        using SVD, reconstructs the matrix with reduced dimensionality, and 
        predicts the ratings. The predicted ratings are clipped to the range [1, 5].
        Args:
            k (int, optional): The number of singular values and vectors to retain 
                during the decomposition. Defaults to 80.
        Returns:
            csr_matrix: A sparse matrix containing the predicted user-movie ratings.
        """
        
        # Decompose the matrix using SVD
        U, sigma, VT = svds(self.zeroed_user_movie, k)
        
        # Convert sigma to a diagonal matrix
        sigma_diag = diags(sigma)
        
        # Reconstruct the ratings matrix
        predicted_ratings = csr_matrix(U) @ sigma_diag @ csr_matrix(VT)
        
        return predicted_ratings.T

    def predict_ratings_by_ALS(self, factors=80, regularization=0.1, iterations=15) -> csr_matrix:
        """
        Predict ratings using Alternating Least Squares (ALS).
        
        :param factors: Number of latent factors.
        :param regularization: Regularization to prevent overfitting.
        :param iterations: Number of iterations for ALS.
        :return: A sparse matrix with predicted ratings.
        """
        # Create an ALS model
        model = AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)

        # Train the model on the transposed matrix (ALS expects an item-user matrix)
        model.fit(self.zeroed_user_movie.T)
        
        # Get predictions for the user-movie matrix
        predicted_ratings = csr_matrix(model.user_factors) @ csr_matrix(model.item_factors.T)

        return predicted_ratings.T

    
    
    def recommend_movies(self, top_n_rated=10, top_n_recommended=5, movie_file="data\\rawdata\\movies.csv"):
        """
        For 10 random users, display the top_n_rated movies they rated the highest
        and top_n_recommended movies recommended by the algorithm, excluding movies
        they have already rated. Movie titles are displayed instead of movie IDs.
        
        :param top_n_rated: Number of top-rated movies to display for each user.
        :param top_n_recommended: Number of recommended movies to display for each user.
        """
        # Select 10 random users
        random_users = np.random.choice(self.real_user_movie.shape[0], 10, replace=False)

        
        self.movie_titles = pd.read_csv(movie_file).set_index('movieId')['title'].to_dict()
        
        for user_index in random_users:
            print(f"User {user_index + 1}:")
            
            # Get the user's actual ratings
            user_ratings = self.real_user_movie[user_index].toarray().flatten()
            top_rated_indices = np.argsort(user_ratings)[-top_n_rated:][::-1]
            print("  Top rated movies:")
            for movie_id in top_rated_indices:
                movie_title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
                print(f"    {movie_title} with rating {user_ratings[movie_id]}")
            
            # Get the user's predicted ratings
            predicted_ratings = self.predicted_user_movie[user_index].toarray().flatten()
            
            # Exclude movies the user has already rated
            unrated_mask = user_ratings == 0
            predicted_ratings[~unrated_mask] = -np.inf  # Set already rated movies to -inf
            
            # Get top recommended movies
            top_recommended_indices = np.argsort(predicted_ratings)[-top_n_recommended:][::-1]
            print("  Recommended movies:")
            for movie_id in top_recommended_indices:
                movie_title = self.movie_titles.get(movie_id, f"Movie {movie_id}")
                print(f"    {movie_title} with predicted rating {predicted_ratings[movie_id]}")
            print()