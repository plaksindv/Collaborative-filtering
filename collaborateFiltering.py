import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances

from scipy.spatial import distance
from math import sqrt


ratings_df = pd.read_csv('ratings.csv')
print('Unique users count: {}'.format(len(ratings_df['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df.shape))

ratings_df.head()

n = 100000
ratings_df_sample = ratings_df[:n]

n_users = len(ratings_df_sample['userId'].unique())
n_movies = len(ratings_df_sample['movieId'].unique())
(n_users, n_movies)

movie_ids = ratings_df_sample['movieId'].unique()

def scale_movie_id(movie_id):
    scaled = np.where(movie_ids == movie_id)[0][0] + 1
    return scaled

ratings_df_sample['movieId'] = ratings_df_sample['movieId'].apply(scale_movie_id)
ratings_df_sample.head()

train_data, test_data = tts(ratings_df_sample, test_size=0.2)

print('Train shape: {}'.format(train_data.shape))
print('Test shape: {}'.format(test_data.shape))

def rmse(prediction, ground_truth):
    prediction = np.nan_to_num(prediction)[ground_truth.nonzero()].flatten()
    ground_truth = np.nan_to_num(ground_truth)[ground_truth.nonzero()].flatten()
    
    mse = mean_squared_error(prediction, ground_truth)
    return sqrt(mse)
train_data_matrix = np.zeros((n_users, n_movies))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    
test_data_matrix = np.zeros((n_users, n_movies))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')

print(distance.cosine([2,2],[1,1])) 
print(distance.cosine([3,3],[2,3]))
print(distance.cosine([3, 3],[1, 1.5]))
print(distance.cosine([3, 3],[1, 3]))

ta = [[5,5,5,0,0], [4,1,0,5,3], [1,0,0,5,0], [5,0,5,0,4]]
pairwise_distances(demo_data, metric='cosine')
def naive_predict(top):
    top_similar_ratings = np.zeros((n_users, top, n_movies))

    for i in range(n_users):
        top_sim_users = user_similarity[i].argsort()[1:top + 1]
        top_similar_ratings[i] = train_data_matrix[top_sim_users]

    pred = np.zeros((n_users, n_movies))
    for i in range(n_users):
        pred[i] = top_similar_ratings[i].sum(axis=0) / top
    
    return pred


def naive_predict_item(top):
    top_similar_ratings = np.zeros((n_movies, top, n_users))

    for i in range(n_movies):
        top_sim_movies = item_similarity[i].argsort()[1:top + 1]
        top_similar_ratings[i] = train_data_matrix.T[top_sim_movies]
        
    pred = np.zeros((n_movies, n_users))
    for i in range(n_movies):
        pred[i] = top_similar_ratings[i].sum(axis=0) / top
    
    return pred.T


print('Method 1:')
naive_pred = naive_predict(7)
print('User-based CF RMSE: ', rmse(naive_pred, test_data_matrix))

naive_pred_item = naive_predict_item(7)
print('Item-based CF RMSE: ', rmse(naive_pred_item, test_data_matrix))

def k_fract_predict(top):
    top_similar = np.zeros((n_users, top))
    
    for i in range(n_users):
        user_sim = user_similarity[i]
        top_sim_users = user_sim.argsort()[1:top + 1]#[-top:]

        for j in range(top):
            top_similar[i, j] = top_sim_users[j]
            
    abs_sim = np.abs(user_similarity)
    pred = np.zeros((n_users, n_movies))
    
    for i in range(n_users):
        indexes = top_similar[i].astype(np.int)
        numerator = user_similarity[i][indexes]
        
        product = numerator.dot(train_data_matrix[indexes])
        
        denominator = abs_sim[i][top_similar[i].astype(np.int)].sum()
        
        pred[i] = product / denominator
    
    return pred


def k_fract_predict_item(top):
    flag = True
    top_similar = np.zeros((n_movies, top))
    
    for i in range(n_movies):
        movies_sim = item_similarity[i]
        top_sim_movies = movies_sim.argsort()[1:top + 1]

        for j in range(top):
            top_similar[i, j] = top_sim_movies.T[j]
            
    abs_sim = np.abs(item_similarity)
    pred = np.zeros((n_movies, n_users))
    
    
    for i in range(n_users):
        indexes = top_similar[i].astype(np.int)
        numerator = item_similarity[i][indexes]
        
        product = numerator.dot(train_data_matrix.T[indexes])
        
        denominator = abs_sim[i][indexes].sum()
        denominator = denominator if denominator != 0 else 1
        
        pred[i] = product / denominator
        
    return pred.T


print('Method 2:')
k_predict = k_fract_predict(7)
print('User-based CF RMSE: ', rmse(k_predict, test_data_matrix))

k_predict_item = k_fract_predict_item(7)
print('Item-based CF RMSE: ', rmse(k_predict_item, test_data_matrix))

def k_fract_mean_predict(top):
    top_similar = np.zeros((n_users, top))
    
    for i in range(n_users):
        user_sim = user_similarity[i]
        top_sim_users = user_sim.argsort()[1:top + 1]

        for j in range(top):
            top_similar[i, j] = top_sim_users[j]
            
    abs_sim = np.abs(user_similarity)
    pred = np.zeros((n_users, n_movies))
    
    for i in range(n_users):
        indexes = top_similar[i].astype(np.int)
        numerator = user_similarity[i][indexes]
        
        mean_rating = np.array([x for x in train_data_matrix[i] if x > 0]).mean()
        diff_ratings = train_data_matrix[indexes] - train_data_matrix[indexes].mean()
        numerator = numerator.dot(diff_ratings)
        denominator = abs_sim[i][top_similar[i].astype(np.int)].sum()
        
        pred[i] = mean_rating + numerator / denominator
        
    return pred

def k_fract_mean_predict_item(top):
    top_similar = np.zeros((n_movies, top))
    
    for i in range(n_movies):
        movie_sim = item_similarity[i]
        top_sim_movies = movie_sim.argsort()[1:top + 1]
        
        for j in range(top):
            top_similar[i, j] = top_sim_movies[j]
    
    abs_sim = np.abs(item_similarity)
    pred = np.zeros((n_movies, n_users))
    
    for i in range(n_movies):
        indexes = top_similar[i].astype(np.int)
        numerator = item_similarity[i][indexes]
        
        mean_rating = np.array([x for x in train_data_matrix.T[i] if x > 0]).mean()
        mean_rating = 0 if np.isnan(mean_rating) else mean_rating
        
        diff_ratings = train_data_matrix.T[indexes] - mean_rating
        numerator = numerator.dot(diff_ratings)
        denominator = abs_sim[i][top_similar[i].astype(np.int)].sum()
        denominator = denominator if denominator != 0 else 1
        
        pred[i] = mean_rating + numerator / denominator
                
    return pred.T


print('Method 3:')
k_predict = k_fract_mean_predict(7)
print('User-based CF RMSE: ', rmse(k_predict, test_data_matrix))

k_predict_item = k_fract_mean_predict_item(7)
print('Item-based CF RMSE: ', rmse(k_predict_item, test_data_matrix))