import logging
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self,users_df,ratings_df):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        self.algo = SVD()
        self.reader = Reader(line_format='user item rating', rating_scale=(1, 5))
        self.users_df = users_df
        self.ratings_df = ratings_df
        self.util_matrix = ratings_df.pivot(index='user',columns='movie',values='rating')
        self.sim_matrix = self.create_sim_matrix_user()
        self.unrated_user_list = self.create_unrated_user_list()
        
    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")

        uir = ratings.drop('timestamp', axis=1)
        data = Dataset.load_from_df(uir, self.reader)
        trainset = data.build_full_trainset()

        self.algo.fit(trainset)

        self.logger.debug("finishing fit")

        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        user = list(requests['user'])
        movie = list(requests['movie'])

        requests = pd.DataFrame(columns=['user', 'movie', 'rating'])
        for i in range(len(user)):
            rate = self.algo.predict(user[i], movie[i])
            requests = requests.append({'user':user[i], 'movie':movie[i], 'rating':rate[3]}, ignore_index=True)

        self.logger.debug("finishing predict")
        return(requests)
    
    def col_to_vector_df(self,df,col_title,index):
        df[col_title] = df[col_title].apply(lambda x: str(x))
        vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b")
        X = vectorizer.fit_transform(df[col_title])

        temp_df = pd.DataFrame(X.toarray())
        temp_df.columns = vectorizer.get_feature_names()
        temp_df.index = df[index]

        return temp_df
    
    def create_sim_matrix_user(self):
        gender_df = self.col_to_vector_df(self.users_df,'Gender','UserID')
        age_df = self.col_to_vector_df(self.users_df,'Age','UserID')
        occupation_df = self.col_to_vector_df(self.users_df,'Occupation','UserID')
        state_df = self.col_to_vector_df(self.users_df,'state','UserID')

        user_feature_df = pd.concat((gender_df,age_df,occupation_df,state_df),axis=1)
        
        user_sim_matrix = pd.DataFrame(cosine_similarity(user_feature_df))
        
        return user_sim_matrix
    
    def create_unrated_user_list(self):
        unrated_users = set(np.array(self.users_df['UserID'])) - set(np.array(self.ratings_df['user']))
        unrated_user_list = list(unrated_users)

        return unrated_user_list
    
    def user_to_rated_user(self, user_list):
        for i in user_list:
            if i not in self.unrated_user_list:
                rated_user_list.append(i)

        return rated_user_list
    
    def get_sim_users(self, user, n=300):
        '''
        '''
        
        user_series = self.sim_matrix.loc[user]

        x = list(user_series.nlargest(n=n+1).index[:])

        x.remove(user)
        
        rated_user_list = self.user_to_rated_user(x)
        
        return rated_user_list
    
    def get_avg_movie_rating(self,user,movie):
        '''
        Calculate AVG Movie rating based on similar users
        
        If no similar user rating, return movie global mean
        '''

        # rated_user_list = self.user_to_rated_user(user_list)
        
        try:
            rated_user_list = self.get_sim_users(user)
            return self.util_matrix[movie].loc[rated_user_list].mean()
        except:
            try:
                return self.util_matrix[movie].mean()
            except:
                return 3.59
            
#         if math.isnan(avg_rating):
#             return self.util_matrix[movie].mean()

#         return avg_rating
        pass

    
    
if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
