import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def imbalance_check(data:pd.DataFrame):
    class_count = data['CONDITION'].value_counts()
    h_count = class_count.loc['H']
    d_count = class_count.loc['D']
    if h_count == d_count:
        return False
    else:
        return True

def scale_data(train, test, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train), columns=list(train.columns))
    test_data = pd.DataFrame(scaler.transform(test), columns=list(train.columns))
    return train_data, test_data

def label_encoding(y):
    return y.replace({'H':1, 'D':0})





