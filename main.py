import os
from feature_selection.data_mngt import read_data, split_data
from feature_selection.data_preprocessing import imbalance_check, label_encoding, scale_data
from feature_selection.models import dtree, rforest, xgboost, perm_knn, chi_2, mutual_inf, categorical_corr, unc_coeff
#from feature_selection.utils import merge_plots

## follow PEP8 standards 
# Class names must be camelcase (Ex: DataManagement)
# function and variable names must be lowercase with words separated by underscore (Ex: read_data, file_path)

if __name__ == '__main__':
    datasets_dir = os.path.join(os.getcwd(), 'Datasets')
    folder_name = "5. PHQ9_GAD7"
    filename = "PHQ9_GAD7_df.csv"
    file_path = os.path.join(datasets_dir, folder_name, filename)
    df = read_data(file_path)

    # checking for class imbalance
    print("imbalanced classes:",imbalance_check(df))

    # splitting the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, threshold=0.8)

    # scaling data using standard scaler by default(use scaler="minmax" for minmax scaling)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # encoding labels into 0 and 1
    y_train_encoded = label_encoding(y_train)
    y_test_encoded = label_encoding(y_test)
    assert y_train.value_counts().loc['H'] == y_train_encoded.value_counts().loc[1]

    # model training for feature selection
    plot_dtree = dtree(X_train_scaled, y_train_encoded)
    print("end of decision tree".center(50,"*"))

    plot_rforest = rforest(X_train_scaled, y_train_encoded)
    print("end of random forest".center(50,'*'))

    plot_xgboost = xgboost(X_train_scaled, y_train_encoded)
    print("end of xgboost".center(50,'*'))

    plot_perm = perm_knn(X_train_scaled, y_train_encoded)
    print("end of permutation importances with knn".center(50,'*'))

    plot_chi2 = chi_2(X_train, y_train_encoded, X_test)
    print("end of chi2 feature selection".center(50,'*'))

    plot_mutualinf = mutual_inf(X_train, y_train_encoded, X_test)
    print("end of mutual information feature selection".center(50,'*'))

    categorical_corr(df)
    print("end of categorical correlation study".center(50,'*'))

    unc_coeff(df)
    print("end of uncertainty coefficients study".center(50,'*'))

    # Merge all different plots in one figure and save it
    #merge_plots(plot_dtree, plot_rforest, plot_xgboost, filename)

    

