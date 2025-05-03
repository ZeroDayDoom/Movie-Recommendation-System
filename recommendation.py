import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv',sep=";")

"""In terms of data quality, I have no duplicates or null values in both of the datasets, so no data treating will be necessary for missing values or duplicates in the next phase.


As the data still need to be prepped for more on detail EDA, I generated two visualisations of the Movies dataset, the first related to words most present on titles and the second a pie chart of genre distribution across titles:
"""

# Split genres and count them
movies["genre_count"] = movies["genres"].apply(lambda x: len(x.split('|')))

# Reorder columns to move 'genre_count' to the front
columns_order = ["genre_count", "movieId", "title", "genres"]
movies = movies[columns_order]

# Extract unique genres
unique_genres = set(movies["genres"].str.split('|').explode())

# Create binary columns for each genre
for genre in unique_genres:
    movies[genre] = movies["genres"].apply(lambda x: 1 if genre in x.split('|') else 0)


import re

# Extract year using regex and create a new column
movies["year"] = movies["title"].apply(lambda x: int(re.search(r"\((\d{4})\)", x).group(1)) if re.search(r"\((\d{4})\)", x) else None)

# Remove the year from the title
movies["title"] = movies["title"].apply(lambda x: re.sub(r"\s*\(\d{4}\)", "", x))
movies["year"] = movies["year"].astype("Int64")  # Keeps NaN values while removing .0

# Count the number of genres for each movie
movies["genre_count"] = movies["genres"].apply(lambda x: len(x.split('|')))

# Move 'genre_count' next to the 'title' column
movies.insert(2, "genre_count", movies.pop("genre_count"))

# Identify movies where genre is missing or not listed properly
movies_without_genre = movies[movies["genres"].isin(["(no genres listed)", "N/A", "None", ""])].copy()

# Remove movies with no listed genre from the dataset
movies = movies[~movies["genres"].isin(["(no genres listed)", "N/A", "None", ""])]

# Reset index after removal
movies.reset_index(drop=True, inplace=True)

# Remove rows where movieId is in movies_without_genre
ratings = ratings[~ratings["movieId"].isin(movies_without_genre["movieId"])]

ratings.drop(columns="timestamp", inplace=True)

# Merging Movies and Rating Datasets on moviesId
merged = movies.merge(ratings, on="movieId", how="inner")
merged = merged.drop(columns=["genres", "(no genres listed)"], errors="ignore")


merged_pivot = merged.pivot_table(values='rating',columns='userId',index='movieId').fillna(0)
print('Shape of this pivot table :',merged_pivot.shape)
merged_pivot.head()

"""#### **Machine Learning Model training for Recommending movies based on users ratings.**"""

from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(merged_pivot)

"""#### **Developing the class of Collaborative filtering Recommendation Engine**"""

class Recommender:
    def __init__(self):
        # This list will stored movies that called atleast ones using recommend_on_movie method
        self.hist = []
        self.ishist = False # Check if history is empty

    # This method will recommend movies based on a movie that passed as the parameter
    def recommend_on_movie(self,movie,n_reccomend = 5):
        self.ishist = True
        movieid = int(movies[movies['title']==movie]['movieId'])
        self.hist.append(movieid)
        distance,neighbors = nn_algo.kneighbors([merged_pivot.loc[movieid]],n_neighbors=n_reccomend+1)
        movieids = [merged_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in [movieid]]
        return recommeds[:n_reccomend]

    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_reccomend = 5):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(merged_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors = nn_algo.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        movieids = [merged_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        return recommeds[:n_reccomend]

# linitializing the Recommender Object
recommender = Recommender()

# Recommendation based on past watched movies, but the object just initialized. So, therefore no history found
recommender.recommend_on_history()
