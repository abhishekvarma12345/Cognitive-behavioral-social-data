import os
from feature_selection.data_mngt import read_data, split_data
from feature_selection.data_preprocessing import imbalance_check

## follow PEP8 standards 
# Class names must be camelcase (Ex: DataManagement)
# function and variable names must be lowercase with words separated by underscore (Ex: read_data, file_path)

if __name__ == '__main__':
    datasets_dir = os.path.join(os.getcwd(), 'Datasets')
    folder_name = "5. PHQ9_GAD7"
    filename = "PHQ9_GAD7_df.csv"
    file_path = os.path.join(datasets_dir, folder_name, filename)
    df = read_data(file_path)

    print("imbalanced classes:",imbalance_check(df))