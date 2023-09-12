import os

ROOT_DIR = os.getcwd()
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m/ml-1m')
import re


if __name__ == "__main__":
    
    # Loading user_dict.npy 
    
    # Specify the path to your .npy file
    file_path = "C:/Users/mpuer/Documents/GitHub/Recommender_system_via_deep_RL_/data/user_dict.npy"

    #   Load the .npy file into a NumPy array
    loaded_array = np.load(file_path,allow_pickle=True)
    a=np.array(loaded_array)
    
    tuple_list = re.findall(r'\((.*?)\)', str(loaded_array))

    # Split each tuple into individual elements and convert them to a NumPy array
    loaded_array = np.array([tuple(map(float, item.split(','))) for item in tuple_list])

    csv_file_path = "C:/Users/mpuer/Documents/GitHub/Recommender_system_via_deep_RL_/data/user_dict.csv"

    # print(loaded_array)
    if loaded_array.size > 0:
        if loaded_array.ndim == 1:
        # If the array is 1D, reshape it to a 2D array with one column
            loaded_array = loaded_array.reshape(-1, 1)
    
        # Create a DataFrame from the NumPy array
        f = pd.DataFrame(loaded_array)
        print(f)
        #    Specify the path where you want to save the CSV file

        # Save the DataFrame as a CSV file using to_csv()
        f.to_csv(csv_file_path, index=False, header=False)  # You can adjust the options as needed
    # file_path = "C:/Users/mpuer/Documents/GitHub/Recommender_system_via_deep_RL_/data/users_histroy_len.npy"

    # #   Load the .npy file into a NumPy array
    # loaded_array = np.load(file_path)
    
    # tuple_list = re.findall(r'\((.*?)\)', str(loaded_array))

    # # Split each tuple into individual elements and convert them to a NumPy array
    # loaded_array = np.array([tuple(map(float, item.split(','))) for item in tuple_list])

    # csv_file_path = "C:/Users/mpuer/Documents/GitHub/Recommender_system_via_deep_RL_/data/users_histroy_len.csv"

    # # print(loaded_array)
    # if loaded_array.size > 0:
    #     if loaded_array.ndim == 1:
    #     # If the array is 1D, reshape it to a 2D array with one column
    #         loaded_array = loaded_array.reshape(-1, 1)
    
    #     # Create a DataFrame from the NumPy array
    #     f = pd.DataFrame(loaded_array)
    #     print(f)
    #     #    Specify the path where you want to save the CSV file

    #     # Save the DataFrame as a CSV file using to_csv()
    #     f.to_csv(csv_file_path, index=False, header=False)  # You can adjust the options as needed

    
    # # ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    # # users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    # # movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
    # # ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = np.uint32)
    # # movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    # # movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    # # print("Data loading complete!")