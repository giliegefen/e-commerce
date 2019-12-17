from collections import defaultdict

import numpy as np
from numpy import array
import pandas as pd
from matplotlib import use
use ('Agg')
from matplotlib import pyplot as plt
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
from numpy.linalg import solve, norm
from numpy import array
from scipy.sparse import coo_matrix

import math

import Timer


class ModelData:
    """The class reads 5 files as specified in the init function, it creates basic containers for the data.
    See the get functions for the possible options, it also creates and stores a unique index for each user and movie
    """

    def __init__(self, train_x, train_y, test_x, test_y, movie_data):
        """Expects 4 data set files with index column (train and test) and 1 income + genres file without index col"""
        self.train_x = pd.read_csv(train_x, index_col=[0])
        self.train_y = pd.read_csv(train_y, index_col=[0])
        self.test_x = pd.read_csv(test_x, index_col=[0])
        self.test_y = pd.read_csv(test_y, index_col=[0])
        self.movies_data = pd.read_csv(movie_data)
        self.users = self._generate_users()
        self.movies = self._generate_movies()
        self.incomes = self._generate_income_dict()
        self.movie_index = self._generate_movie_index()
        self.user_index = self._generate_user_index()

    def _generate_users(self):
        users = sorted(set(self.train_x['user']))
        return tuple(users)

    def _generate_movies(self):
        movies = sorted(set(self.train_x['movie']))
        return tuple(movies)

    def _generate_income_dict(self):
        income_dict = defaultdict(float)
        average_income = np.float64(self.movies_data['income'].mean())
        movies = sorted(set(self.movies_data['movie']))
        for m in movies:
            movie_income = int(self.movies_data[self.movies_data['movie'] == m]['income'])
            income_diff = movie_income * 10e-10
            income_dict[m] = income_diff
        return income_dict

    def _generate_user_index(self):
        user_index = defaultdict(int)
        for i, u in enumerate(self.users):
            user_index[u] = i
        return user_index

    def _generate_movie_index(self):
        movie_index = defaultdict(int)
        for i, m in enumerate(self.movies, start=len(self.users)):
            movie_index[m] = i
        return movie_index

    def get_users(self):
        """:rtype tuples of all users"""
        return self.users

    def get_movies(self):
        """:rtype tuples of all movies"""
        return self.movies

    def get_movie_index(self, movie):
        """:rtype returns the index of the movie if it exists or None"""
        return self.movie_index.get(movie, None)

    def get_user_index(self, user):
        """:rtype returns the index of the user if it exists or None"""
        return self.user_index.get(user, None)

    def get_movie_income(self, movie):
        """:rtype returns the income of the movie if it exists or None"""
        if self.incomes.get(movie, None) is None:
            print(movie)
        return self.incomes.get(movie, None)

    def get_movies_for_user(self, user):
        return self.train_x[self.train_x['user'] == user]['movie'].values

    def get_users_for_movie(self, movie):
        return self.train_x[self.train_x['movie'] == movie]['user'].values

    def get_indexes_for_user(self):
        indexes=defaultdict(list)
        for i, row in self.train_x.iterrows():
            indexes[row[0]].append(i)
        return indexes

    def get_indexes_for_movies(self):
        indexes=defaultdict(list)
        for i, row in self.train_x.iterrows():
            indexes[row[1]].append(i)
        return indexes


def create_coefficient_matrix(train_x, data: ModelData = None):
    matrix_timer = Timer.Timer('Matrix A creation')
    # Modify this function to return the coefficient matrix A as seen in the lecture (slides 24 - 37).
    train_x['user_index'] = train_x['user'].apply(data.get_user_index)
    train_x['movie_index'] = train_x['movie'].apply(data.get_movie_index)

    n_matrix = len(train_x.values)
    row = np.arange(n_matrix)


    users_ind = train_x['user_index'].values
    movies_ind = train_x['movie_index'].values

    base = np.ones(len(train_x))
    base[0] += (0.001 * np.random.random())
    users_spr = coo_matrix((base, (row, users_ind)), shape=(n_matrix, n_matrix))


    base[0] += (0.001 * np.random.random())
    movies_spr = coo_matrix((base, (row, movies_ind)), shape=(n_matrix, n_matrix))
    matrix_a = users_spr + movies_spr

    return matrix_a



def create_coefficient_matrix_with_income(train_x, data: ModelData = None):
    matrix_timer = Timer.Timer('Matrix A with income creation')
    print('start matric with income')
    # TODO: Modify this function to return a coefficient matrix A for the new model with income

    train_x['user_index'] = train_x['user'].apply(data.get_user_index)
    train_x['movie_index'] = train_x['movie'].apply(data.get_movie_index)
    train_x['movie_income'] = train_x['movie'].apply(data.get_movie_income)

    income = train_x['movie_income'].values


    n_matrix = len(train_x.values)
    row = np.arange(n_matrix)

    users_ind = train_x['user_index'].values
    movies_ind = train_x['movie_index'].values

    base = np.ones(len(train_x))
    base[0] += (0.001 * np.random.random())
    users_spr = coo_matrix((base, (row, users_ind)), shape=(n_matrix, n_matrix+1))
    noise0= (0.001 * np.random.random())
    base[0] += noise0
    movies_spr = coo_matrix((base, (row, movies_ind)), shape=(n_matrix, n_matrix+1))
    matrix_a = users_spr + movies_spr

    income_ind = np.full(n_matrix, n_matrix)
    noise=(0.001 * np.random.random())
    income[0] += noise
    income_spr = coo_matrix((income, (row, income_ind)), shape=(n_matrix, n_matrix+1))
    matrix_a += income_spr

    matrix_timer.stop()
    return matrix_a

def construct_rating_vector(train_y, r_avg):
    # Modify this function to return vector C as seen in the lecture (slides 24 - 37).
    c = np.array(train_y['rate'].values) - r_avg
    return c


def fit_parameters(matrix_a, vector_c):
    a = coo_matrix(matrix_a).tocsr()
    ata = a.T.dot(a)
    atc = a.T.dot(vector_c)
    b = lsqr(ata, atc)
    return b[0]


def calc_parameters(r_avg, train_x, train_y, data: ModelData = None):
    # Modify this function to return the calculated average parameters vector b (slides 24 - 37).
    users = data.get_users()
    movies = data.get_movies()
    b=[]
    indexes= data.get_indexes_for_user()
    for u in users:
        sumr=0
        for i in indexes[u]:
            rate= train_y.loc[i,'rate']
            sumr=sumr+rate
        u_avg=sumr/len(indexes[u])
        b.append(u_avg-r_avg)

    indexes_m= data.get_indexes_for_movies()

    for m in movies:
        sumrm=0
        for ik in indexes_m[m]:
            rate= train_y.loc[ik,'rate']
            sumrm=sumrm+rate
        i_avg=sumrm/len(indexes_m[m])
        b.append(i_avg-r_avg)

    b_array= np.asarray(b)
    return b_array


def calc_average_rating(train_y):
    #  Modify this function to return the average rating r_avg.
    #r_avg = train_y.mean(axis=0)['rate']
    sum=0
    for entry in train_y.iterrows():
        if entry[0]!= 'rate':
            sum=sum+float(entry[1])
    leng= len(train_y)
    r_avg = (sum/leng)
    return r_avg


def model_inference(test_x, vector_b, r_avg, data: ModelData = None):
    # TODO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    predictions_list = []
    users=data.get_users()
    movies=data.get_movies()
    for i,row in test_x.iterrows():
        if row[0]!='user' or row[0]!='rate':
            bu=0
            bi=0
            if (row[0] in users):
                bu = vector_b[users.index(row[0])]
                bi=bi
            if (row[1] in movies):
                tr=(len(users))+movies.index(row[1])

                bi = vector_b[tr]##
                bu=bu


        predictions_list += [r_avg+bu+bi]

    return predictions_list


def model_inference_with_income(test_x, vector_b, r_avg, data: ModelData = None):
    # TODO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    # TODO: based on the modified model with income
    predictions_list = []

    users = data.get_users()
    movies = data.get_movies()
    for i, row in test_x.iterrows():
        if row[0] != 'user' or row[0] != 'rate':
            u_index = data.get_user_index(row[0])
            m_income = data.get_movie_income(row[1])
            if (u_index==None) and (m_income==None):
                predictions_list += [r_avg]
            elif (m_income==None):
                predictions_list += [vector_b.item(u_index)]
            elif (u_index==None):
                predictions_list += [r_avg+(vector_b[-1])*m_income]
            else:
                predictions_list += [r_avg +vector_b.item(u_index)+(vector_b[-1])*m_income]

    return predictions_list


def calc_error(predictions_df, test_df):
    #  Modify this function to return the RMSE
    rmse=0

    for i,row in predictions_df.iterrows():
        rmse += ((row[2]-test_df.loc[i, 'rate'])**2)
    rmse = math.sqrt(rmse/len(predictions_df))
    return rmse


def calc_avg_error(predictions_df, test_df):
    #  Modify this function to return a dictionary of tuples {MOVIE_ID:(RMSE, RATINGS)}
    m_error = defaultdict(tuple)

    rmse = defaultdict(list)

    for i, row in predictions_df.iterrows():
        rmse[row[1]] += [((row[2] - test_df.loc[i, 'rate']) ** 2)]

    for m in rmse:
        rmse_divided = math.sqrt(sum(rmse[m]) / len(rmse[m]))
        m_error[m]=(rmse_divided,len(rmse[m]))

    # In this example movie 3 was ranked by 5 users and has an RMSE of 0.9
    # m_error[3] = (0.9, 5)
    #print("m_error:", m_error)
    return m_error


def plot_error_per_movie(movie_error):
    # TODO: Modify this function to plot the graph y:(RMSE of movie i) x:(number of users rankings)
    to_plot= np.array(list(movie_error.values()))
    y,x = to_plot.T
    plt.scatter(x,y)
    plt.show()

""" directions: https://www.digitalocean.com/community/tutorials/how-to-plot-data-in-python-3-using-matplotlib """


