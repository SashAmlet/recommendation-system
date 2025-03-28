import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, diags
from scipy.sparse.linalg import svds
from datetime import datetime
# import cupy as cp

class MovieRecommender:
    def __init__(self, real_ratings, modified_ratings):
        self.predicted_user_movie = None
        self.real_user_movie = csr_matrix(
                (real_ratings['rating'], (real_ratings['userId'], real_ratings['movieId']))
            )
        
        self.zeroed_user_movie = csr_matrix(
                (modified_ratings['rating'], (modified_ratings['userId'], modified_ratings['movieId']))
            )

    
    def train(self, method="MatrixFactorization", generate_new=False, filename="data\\predicted_ratings_MatrixFactorization.npz"):
        if not generate_new:
            self.predicted_user_movie = load_npz(filename)
        else:
            if method == "MatrixFactorization":
                self.predicted_user_movie = self.predict_ratings_by_MatrixFactorization()
            # elif method == "MCMC":
            #     self.predicted_user_movie = self.predict_ratings_by_MCMC()
            else:
                raise ValueError("Unsupported method. Choose 'MatrixFactorization' or 'MCMC'.")

            # Generate filename with method and date
            date_str = datetime.now().strftime("%Y%m%d")
            save_filename = f"data\\predicted_ratings_{method}_{date_str}.npz"
            save_npz(save_filename, self.predicted_user_movie)
        

    def predict_ratings_by_MatrixFactorization(self, k=50) -> csr_matrix:
        """Предсказание оценок с использованием матричного разложения для csr_matrix"""
        
        # Разложение матрицы с помощью SVD
        U, sigma, VT = svds(self.zeroed_user_movie, k)
        
        # Преобразование sigma в диагональную матрицу
        sigma_diag = diags(sigma)
        
        # Восстановление матрицы оценок
        predicted_ratings = csr_matrix(U) @ sigma_diag @ csr_matrix(VT)
        
        return predicted_ratings

    
    
    def recommend_movies(self, user_id, top_n=10):
        user_index = user_id - 1  # Assuming userId starts from 1
        predicted_ratings = self.predict_ratings()
        recommendations = pd.Series(predicted_ratings[user_index]).sort_values(ascending=False).head(top_n)
        return recommendations