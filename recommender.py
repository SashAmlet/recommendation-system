import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, diags
from scipy.sparse.linalg import svds
# import cupy as cp

class MovieRecommender:
    def __init__(self, ratings, modified_ratings):
        self.predicted_user_movie = None
        self.real_user_movie = csr_matrix(
                (ratings['rating'], (ratings['userId'], ratings['movieId']))
            )
        self.zeroed_user_movie = csr_matrix(
                (modified_ratings['rating'], (modified_ratings['userId'], modified_ratings['movieId']))
            )

    def covariance_matrix_with_missing(self, matrix: csr_matrix):
        """
        Compute the covariance matrix for sparse data with missing values.
        """
        n_cols = matrix.shape[1]
        cov_matrix = np.zeros((n_cols, n_cols))

        for i in range(n_cols):
            for j in range(n_cols):
                col_i = matrix.getcol(i).toarray().flatten()
                col_j = matrix.getcol(j).toarray().flatten()

                # Identify non-missing values
                valid_indices = (col_i != 0) & (col_j != 0)
                if np.count_nonzero(valid_indices) > 1:
                    valid_data_i = col_i[valid_indices]
                    valid_data_j = col_j[valid_indices]
                    cov_matrix[i, j] = np.cov(valid_data_i, valid_data_j, bias=False)[0, 1]
                else:
                    cov_matrix[i, j] = 0  # Handle cases with insufficient data

        return cov_matrix

    def find_missing_data(self, matrix: csr_matrix):
        """
        Identify missing data in a sparse matrix.
        """
        missing_data_info = {}
        for row_idx in range(matrix.shape[0]):
            row = matrix.getrow(row_idx).toarray().flatten()
            missing_indices = np.where(row == 0)[0]  # Missing values are zeros
            if len(missing_indices) > 0:
                missing_data_info[row_idx] = missing_indices.tolist()
        return missing_data_info

    def predict_ratings_by_MCMC(self, matrix: csr_matrix, numOfIterations=50):
        """
        Perform MCMC sampling on sparse data represented as a csr_matrix.
        """
        n_rows, n_cols = matrix.shape
        mean_vector = np.zeros(n_cols)
        for col_idx in range(n_cols):
            col = matrix.getcol(col_idx).toarray().flatten()
            non_zero_values = col[col != 0]
            mean_vector[col_idx] = np.mean(non_zero_values) if len(non_zero_values) > 0 else 0

        covariance_matrix = self.covariance_matrix_with_missing(matrix)
        indexes_with_nan = self.find_missing_data(matrix)

        filled_matrix = matrix.copy()

        for _ in range(numOfIterations - 1):
            for row, cols in indexes_with_nan.items():
                old_approximation = filled_matrix.getrow(row).toarray().flatten()
                new_approximation = old_approximation.copy()

                for col in cols:
                    std_dev = np.sqrt(covariance_matrix[col, col]) if covariance_matrix[col, col] > 0 else 1
                    new_approximation[col] = old_approximation[col] + np.random.normal(0, std_dev)

                # M-step
                if (
                    multivariate_normal.pdf(new_approximation, mean_vector, covariance_matrix)
                    / multivariate_normal.pdf(old_approximation, mean_vector, covariance_matrix)
                    > np.random.uniform(0, 1)
                ):
                    for col in cols:
                        filled_matrix[row, col] = new_approximation[col]

        return filled_matrix
    
    def train(self, generate_new=False, filename=None):
        if not generate_new:
            self.predicted_user_movie = load_npz(filename)
        else:

            self.predicted_user_movie = self.predict_ratings_by_MatrixFactorization() # self.predict_ratings_by_MCMC(user_movie_matrix)
            save_npz("data\\data_predicted_ratings_10k.npz", self.predicted_user_movie)
        

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