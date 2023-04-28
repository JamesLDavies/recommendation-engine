import toml
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors


CONFIG_PATH = r"C:\Users\james\PycharmProjects\recommendation-engine\config.toml"

config = toml.load(CONFIG_PATH)


if __name__ == "__main__":
    movies_path = config['paths']['movies']
    movies_df = pd.read_csv(movies_path)

    ratings_path = config['paths']['ratings']
    ratings_df = pd.read_csv(ratings_path)

    # Collaborative Filtering
    print(ratings_df.head())

    # Temp fix for pivot being too large
    ratings_chunk_df = ratings_df.iloc[:5000]

    ratings_pivot_df = (ratings_chunk_df
                        .pivot_table(values='rating',
                                     columns='userId',
                                     index='movieId')
                        .fillna(0))
    print(ratings_pivot_df.head())

    nn_algo = NearestNeighbors(metric='cosine')
    nn_algo.fit(ratings_pivot_df)

    # Build a Recommender class
    class Recommender:
        def __init__(self):
            self.hist = []
            self.ishist = False

        def recommend_on_movie(self, movie: str, n_recommend: int=5) -> tuple:
            """Recommend movies based on a movie that passed as the parameter"""
            self.ishist = True
            movie_id = int(movies_df[movies_df['title']==movie]['movieId'])
            self.hist.append(movie_id)
            (dist,
             neighbours) = nn_algo.kneighbors([ratings_pivot_df.loc[movie_id]],
                                              n_neighbors=n_recommend+1)
            movie_ids = [ratings_pivot_df.iloc[i].name for i in neighbours[0]]
            recommends = [
                str(movies_df[movies_df['movieId']==m_id]['title']).split('\n')[0].split('\t')[-1]
                for m_id in movie_ids if m_id not in [movie_id]]
            return recommends[:n_recommend]

        def recommend_on_history(self, n_recommend: int=5) -> tuple:
            """Recommend movies based on history that is stored in self.hist"""
            # TODO: Have this return a proper error
            if not self.ishist:
                return print('No history')
            hist = np.array([list(ratings_pivot_df.loc[m_id]) for m_id in self.hist])
            (dist, neighbours) = nn_algo.kneighbors([np.average(hist, axis=0)], n_neighbors=n_recommend+len(self.hist))
            movie_ids = [ratings_pivot_df.iloc[i].name for i in neighbours[0]]
            recommends = [str(movies_df[movies_df['movieId']==m_id]['title']).split('\n')[0].split('  ')[-1] for m_id in movie_ids if m_id not in self.hist]
            return recommends[:n_recommend]


    recommender = Recommender()
    print(recommender.recommend_on_movie('Father of the Bride Part II (1995)'))

    print(recommender.recommend_on_history())