import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_path):
    df = pd.read_csv(file_path,sep=";")
    return df

def split_data(data:pd.DataFrame):
    pass




    

