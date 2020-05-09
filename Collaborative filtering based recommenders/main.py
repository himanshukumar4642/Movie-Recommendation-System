# http://surprise.readthedocs.io/en/stable/getting_started.html
# I believe in loading all the datasets from pandas df
# you can also load dataset from csv and whatever suits

ratings = pd.read_csv('ratings_small.csv') # reading data in pandas df

from surprise import Reader, Dataset

# to load dataset from pandas df, we need `load_fromm_df` method in surprise lib

ratings_dict = {'itemID': list(ratings.movieId),
                'userID': list(ratings.userId),
                'rating': list(ratings.rating)}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is required.
# The Reader class is used to parse a file containing ratings.
reader = Reader(rating_scale=(0.5, 5.0))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

# Split data into 5 folds

data.split(n_folds=5)

from surprise import SVD, evaluate
from surprise import NMF

# svd
algo = SVD()
evaluate(algo, data, measures=['RMSE'])

# nmf
algo = NMF()
evaluate(algo, data, measures=['RMSE'])