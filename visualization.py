import matplotlib.pyplot as plt
import seaborn as sns

class MovieDataVisualizer:
    def __init__(self, ratings, movies):
        """
        Initialize the visualizer with ratings and movies data.
        """
        self.ratings = ratings
        self.movies = movies

    def plot_top_rated_movies(self):
        """Visualization of the top 10 movies with the highest number of ratings"""
        # Count the number of ratings for each movie
        movie_rating_counts = self.ratings.groupby("movieId")["rating"].count().reset_index()
        movie_rating_counts = movie_rating_counts.merge(self.movies, on="movieId")
        top_rated_movies = movie_rating_counts.sort_values(by="rating", ascending=False).head(10)

        # Visualization
        plt.figure(figsize=(10, 5))
        sns.barplot(y=top_rated_movies["title"], x=top_rated_movies["rating"], palette="viridis")
        plt.xlabel("Number of Ratings")
        plt.ylabel("Movie")
        plt.title("Movies with the Highest Number of Ratings")
        plt.tight_layout()  # Ensure full labels are visible
        plt.show()

    def plot_movie_ratings(self):
        """Movies with the highest and lowest ratings."""
        # Calculate the average rating for each movie
        movie_avg_ratings = self.ratings.groupby("movieId")["rating"].mean().reset_index()
        movie_avg_ratings = movie_avg_ratings.merge(self.movies, on="movieId")

        # Count the number of ratings
        movie_rating_counts = self.ratings.groupby("movieId")["rating"].count()
        
        # Calculate the mean and standard deviation
        mean_ratings = movie_rating_counts.mean()
        std_ratings = movie_rating_counts.std()
        
        # Adjust the threshold: it must be at least 100 but not lower than (mean - std)
        min_ratings_threshold = max(mean_ratings - std_ratings, 100)

        # Set the index for movie_avg_ratings
        movie_avg_ratings = movie_avg_ratings.set_index("movieId")
        # Filter by the number of ratings
        filtered_movies = movie_avg_ratings[movie_rating_counts > min_ratings_threshold].reset_index()

        # Top 10 movies with the highest ratings
        top_movies = filtered_movies.sort_values(by="rating", ascending=False).head(10)

        # Top 10 movies with the lowest ratings
        worst_movies = filtered_movies.sort_values(by="rating", ascending=True).head(10)

        # Visualization
        _, ax = plt.subplots(1, 2, figsize=(15, 5))

        sns.barplot(y=top_movies["title"], x=top_movies["rating"], ax=ax[0], palette="Blues")
        ax[0].set_title("Movies with the Highest Ratings")

        sns.barplot(y=worst_movies["title"], x=worst_movies["rating"], ax=ax[1], palette="Reds")
        ax[1].set_title("Movies with the Lowest Ratings")

        plt.tight_layout()  # Ensure full labels are visible
        plt.show()

    def plot_rating_distribution(self):
        """Distribution of user ratings"""
        plt.figure(figsize=(10, 5))

        # Use a subset if the data exceeds 100,000
        if len(self.ratings) > 100_000:
            sample_data = self.ratings["rating"].sample(100_000, random_state=42)
        else:
            sample_data = self.ratings["rating"]

        # Split data into integers and fractional parts
        integer_ratings = sample_data[sample_data % 1 == 0]
        fractional_ratings = sample_data[sample_data % 1 != 0]

        # Visualization on a single plot
        sns.histplot(integer_ratings, bins=20, kde=False, color="green", label="Integer Ratings")
        sns.histplot(fractional_ratings, bins=20, kde=False, color="orange", label="Fractional Ratings")

        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.tight_layout()  # Ensure full labels are visible
        plt.show()

    def plot_top_users(self):
        """Top 10 users by the number of ratings"""
        user_rating_counts = self.ratings.groupby("userId")["rating"].count().reset_index()
        top_users = user_rating_counts.sort_values(by="rating", ascending=False).head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_users["userId"], y=top_users["rating"], palette="viridis", order=top_users["userId"])
        plt.xlabel("User")
        plt.ylabel("Number of Ratings")
        plt.title("Top 10 Users by Number of Ratings")
        plt.tight_layout()  # Ensure full labels are visible
        plt.show()

    def plot_user_types(self):
        """Clustering users by rating behavior"""
        # Define user groups based on ratings
        def classify_user(user_ratings):
            n = 10
            positive = sum(user_ratings >= 4)
            negative = sum(user_ratings <= 2)
            if positive > negative * n:
                return "Positive"
            elif negative > positive * n:
                return "Negative"
            return "Neutral"

        user_types = self.ratings.groupby("userId")["rating"].apply(classify_user)
        user_types_counts = user_types.value_counts()

        # Visualization using a bar chart
        plt.figure(figsize=(8, 5))
        sns.barplot(
            x=user_types_counts.index,
            y=user_types_counts.values,
            palette=["red", "gray", "blue"]
        )
        plt.xlabel("User Type")
        plt.ylabel("Number of Users")
        plt.title("Distribution of Users by Rating Behavior")
        plt.tight_layout()  # Ensure full labels are visible
        plt.show()
