#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from recommender import DRRAgent

import os
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'src/Servicio/app/DRL_model_2/data/ml-1m')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10

# os.environ["CUDA_VISIBLE_DEVICES"]="1"  

if __name__ == "__main__":

    print('Data loading...')
    print('ROOT_DIR: ', ROOT_DIR)
    print('ROOT_DIR: ', DATA_DIR)
    #Loading datasets
    
    ratings_list = [ i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]

    ratings_list_int =[[int(x),int(y),int(z),int(t)] for [x,y,z,t] in  ratings_list]

    ratings_df = pd.DataFrame(ratings_list_int, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.float64)
    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # movie id as movie title
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list} #PAra la evaluación
    #{'1': ['Toy Story (1995)', "Animation|Children's|Comedy"], '2': ['Jumanji (1995)', "Adventure|Children's|Fantasy"], ....}
    ratings_df = ratings_df.applymap(int)

    # Organize movies viewed by user in order
    users_dict = np.load(os.path.join(ROOT_DIR, 'src/Servicio/app/DRL_model_2/data/user_dict.npy'), allow_pickle=True)
    #  1: [(3186, 4), (1721, 4), (1270, 5), (1022, 5), (2340, 3), (1836, 5), (3408, 4), (1207, 4), (2804, 5), (260, 4), (720, 3), (1193, 5), (919, 4), (608, 4), (2692, 4), (1961, 5), (2028, 5), (3105, 5), (938, 4), (1035, 5), (1962, 4), (1028, 5), (2018, 4), (150, 5), (1097, 4), (914, 3), (1287, 5), (2797, 4), (1246, 4), (2762, 4), (661, 3), (2918, 4), (531, 4), (3114, 4), (2791, 4), (1029, 5), (2321, 3), (1197, 3), (594, 4), (2398, 4), (1545, 4), (527, 5), (745, 3), (595, 5), (588, 4), (1, 5), (2687, 3), (783, 4), (2294, 4), (2355, 5), (1907, 4), (1566, 4), (48, 5)],
    #2: [(1198, 4), (1217, 3), (1210, 4), (2717, 3), (1293, 5), (2943, 4), (1225, 5), (1193, 5), (318, 5), (2858, 4), (3030, 4), (1213, 2), (1945, 5), (1207, 4), (3095, 4), (593, 5), (3468, 5), (515, 5), (1873, 4), (1090, 2), (2501, 5), (110, 5), (3035, 4), (2067, 5), (3147, 5), (1247, 5), (3105, 4), (1357, 5), (1196, 5), (1957, 5), (920, 5), (1953, 4), (1834, 4), (1084, 3), (1962, 5), (3735, 3), (3654, 3), (3471, 5), (1784, 5), (1954, 5), (1259, 5), (2728, 3), (1968, 2), (1103, 3), (902, 2), (3451, 4), (2852, 3), (3334, 4), (3578, 5), (3068, 4), (265, 4), (2312, 3), (590, 5), (1253, 3), (3071, 4), (1244, 3), (3699, 2), (1245, 2), (1955, 4), (2236, 5), (3678, 3), (982, 4), (2194, 4), (1442, 4), (2268, 5), (3255, 4), (235, 3), (647, 3), (1096, 4), (1246, 5), (498, 3), (1124, 5), (3893, 1), (1188, 4), (1537, 4), (2396, 4), (2359, 3), (2321, 3), (356, 5), (3108, 3), (1265, 3), (3809, 3), (457, 4), (589, 4), (2571, 4), (2028, 4), (163, 4), (380, 5), (2916, 3), (3418, 4), (1610, 5), (480, 5), (349, 4), (1527, 4), (1408, 3), (3256, 2), (21, 1), (2006, 3), (2353, 4), (1370, 5), (2278, 3), (648, 4), (2427, 2), (1552, 3), (1372, 3), (1792, 3), (2490, 3), (1385, 3), (780, 3), (2881, 3), (165, 3), (1801, 3), (368, 4), (3107, 2), (459, 3), (1597, 3), (442, 3), (2628, 3), (1690, 3), (3257, 3), (2002, 5), (736, 4), (2126, 3), (292, 3), (95, 2), (1917, 3), (1544, 4), (434, 2), (1687, 3)],
    # users_dict -> { id_user:[(id_movie, rating), (id_movie, rating), ...], id_user:[(id_movie, rating), (id_movie, rating), ...], ...)]}
    
    # movie history length for each usersrc/Servicio/app/DRL_model_2/data/users_histroy_len.
    a=os.path.join(ROOT_DIR, 'src/Servicio/app/DRL_model_2/data/users_histroy_len.npy')
    # users_history_lens = np.load(a, allow_pickle=True)

    users_history_lens = [int(i.strip()) for i in open(a,encoding='latin-1').readlines()]
    # Es el cadinal de cuantas veces el usuario a puesto una peli con una valoracion mayor o igual que 4. 
    
    # TODO! ??  creo que me faltaria una cosa y es el diccionario de users_dict pero con las interacciones de ranking mayor o igual que 4
    
    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # Training setting
    train_users_num = int(users_num * 0.8) # 80% of users for training
    train_items_num = items_num # all items for training
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    # print(train_users_dict)
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print('DONE!')
    time.sleep(2)
    user_dict_array=  {k:users_dict.item().get(k) for k in ratings_df['UserID'].unique().tolist()}
    for i in range(1, 10):
        print('user_dict_array: ', len(user_dict_array[i]))

    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)
    