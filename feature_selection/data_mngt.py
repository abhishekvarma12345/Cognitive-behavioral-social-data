import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# setting seed reproducibility in case of randomness(need to be moved to constants.yml file)
SEED = 23

def read_data(file_path):
    """
    Read data from a csv file
    Parameters:
    - file_path (str): path to the csv file
    Returns:
    - DataFrame : Dataframe containing the data
    """
    df = pd.read_csv(file_path,sep=";")
    return df

def split_data(data:pd.DataFrame, threshold=0.8):
    """
    Split data into train and test sets
    Parameters:
    - data (DataFrame): Dataframe containing the dataset
    - threshold (float): Threshold to split the data. Default is 0.8
    Returns:
    - X_train, X_test, y_train, y_test : Splitted data
    """
    X = data.drop(["CONDITION"], axis=1)
    y = data['CONDITION']

    # splitting the classes in train and test sets equally
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=threshold, random_state=SEED, stratify=y)
    return X_train, X_test, y_train, y_test








    

