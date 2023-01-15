import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.combine import SMOTETomek

def imbalance_check(data:pd.DataFrame):
    """
    Check if the dataset is imbalanced.
    Parameters:
    - data (DataFrame): Dataframe containing the dataset
    Returns:
    - bool: True if the dataset is imbalanced, False otherwise.
    """
    class_count = data['CONDITION'].value_counts()
    h_count = class_count.loc['H']
    d_count = class_count.loc['D']
    if h_count == d_count:
        return False
    else:
        return True

def scale_data(train, test, scaler=None):
    """
    Scale features using a scaler
    Parameters:
    - train (DataFrame): Training Data
    - test (DataFrame): Testing Data
    - scaler (object): Scaler object to use. Default is None.
    Returns:
    - train_data (DataFrame): Scaled training Data
    - test_data (DataFrame): Scaled testing Data
    """
    if scaler == None:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train), columns=list(train.columns))
    test_data = pd.DataFrame(scaler.transform(test), columns=list(train.columns))
    return train_data, test_data

def label_encoding(y):
    """
    Perform label encoding on target variable
    Parameters:
    - y (Series or array-like): Target variable.
    Returns:
    - y (Series or array-like): Encoded target variable.
    """
    return y.replace({'H':1, 'D':0})


def smote_tomek(X_train, y_train):
    """
    Perform oversampling with the SMOTE technique and undersampling with the Tomek technique to balance the dataset.
    Parameters:
    - X_train (array-like): Training features
    - y_train (array-like): Training target variable
    Returns:
    - X_train (array-like): Balanced training features
    - y_train (array-like): Balanced training target variable
    """
    print("Balancing with Smote-Tomek...")
    smt = SMOTETomek(sampling_strategy='auto')
    X_train, y_train = smt.fit_resample(X_train, y_train)
    return X_train, y_train





