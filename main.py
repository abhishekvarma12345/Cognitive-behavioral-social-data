import os
from feature_selection.data_mngt import read_data, split_data
from feature_selection.data_preprocessing import imbalance_check, label_encoding, scale_data
from feature_selection.models import dtree, rforest, xgboost, perm_knn, chi_2, mutual_inf, categorical_corr, unc_coeff
from feature_selection.utils import princ_comp_anal
from feature_selection.utils import merge_plots
from feature_selection.utils import make_timestamp_dir

## follow PEP8 standards 
# Class names must be camelcase (Ex: DataManagement)
# function and variable names must be lowercase with words separated by underscore (Ex: read_data, file_path)

if __name__ == '__main__':
    folders_and_files = [("1. shortDT_1","DT_df_CC.csv"),("1. shortDT_2", "DT_df_JI.csv"),("2. PRMQ", "PRMQ_df.csv" ),
                       ("3. PCL", "PCL5_df.csv"),("4. NAQ_R", "NAQ_R_df.csv"),("5. PHQ9_GAD7", "PHQ9_GAD7_df.csv"),
                       ("6. PID5", "PID5_df.csv"),("7. shortPID5", "sPID-5_df.csv"),("8. PRFQ", "PRFQ_df.csv"),
                       ("9. IESR", "IESR_df.csv"),("10. R_NEO_PI", "faked_honest_combined.csv"),
                       ("11. DDDT", "RAW_DDDT.CSV"),("12. IADQ", "IADQ_df.csv"),("13. BF_1", "BF_df_CTU.csv"), 
                       ("13. BF_2", "BF_df_OU.csv"), ("13. BF_3", "BF_df_V.csv")]
    datasets_dir = os.path.join(os.getcwd(), 'Datasets')
    folder_name = folders_and_files[5][0]
    filename = folders_and_files[5][1]
    file_path = os.path.join(datasets_dir, folder_name, filename)
    df = read_data(file_path)

    # Make a time stamp directory where storing the plots
    mydir = make_timestamp_dir(folder_name)

    # checking for class imbalance
    print("Imbalanced classes:",imbalance_check(df))

    # splitting the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, threshold=0.8)

    # scaling data using standard scaler by default(use scaler="minmax" for minmax scaling)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # encoding labels into 0 and 1
    y_train_encoded = label_encoding(y_train)
    y_test_encoded = label_encoding(y_test)
    assert y_train.value_counts().loc['H'] == y_train_encoded.value_counts().loc[1]

    # model training for feature selection
    plot_dtree = dtree(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, mydir)
    print("end of decision tree".center(50,"*"))

    plot_rforest = rforest(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, mydir)
    print("end of random forest".center(50,'*'))

    plot_xgboost = xgboost(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, mydir)
    print("end of xgboost".center(50,'*'))

    plot_perm = perm_knn(X_train_scaled, y_train_encoded, mydir)
    print("end of permutation importances with knn".center(50,'*'))

    plot_chi2 = chi_2(X_train, y_train_encoded, X_test, mydir)
    print("end of chi2 feature selection".center(50,'*'))

    plot_mutualinf = mutual_inf(X_train, y_train_encoded, X_test, mydir)
    print("end of mutual information feature selection".center(50,'*'))

    categorical_corr(df, mydir)
    print("end of categorical correlation study".center(50,'*'))

    unc_coeff(df, mydir)
    print("end of uncertainty coefficients study".center(50,'*'))

    print("Start reduction of dimensionality using PCA".center(50,'*'))
    X_pca = princ_comp_anal(X_train_scaled, mydir)

    # Merge all different plots in one figure and save it
    artifact_dir = os.path.join(os.getcwd(), 'feature_selection', 'artifacts')
    dataset_name = '5. PHQ9_GAD7'
    time_stamp = os.listdir(os.path.join(artifact_dir, dataset_name))[-1]
    plots_dir = os.path.join(artifact_dir, dataset_name, time_stamp)
    merge_plots(plots_dir , "combined.png")

    

