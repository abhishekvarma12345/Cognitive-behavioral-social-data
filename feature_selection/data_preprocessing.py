import pandas as pd
from sklearn.preprocessing import StandardScaler

def imbalance_check(data:pd.DataFrame):
    class_count = data['CONDITION'].value_counts()
    h_count = class_count.loc['H']
    d_count = class_count.loc['D']
    if h_count == d_count:
        return False
    else:
        return True

    

def scale_data(data:pd.DataFrame)->pd.DataFrame:
    pass


