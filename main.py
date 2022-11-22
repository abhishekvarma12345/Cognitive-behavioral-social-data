import os
import argparse
from feature_selection.logger import logging
from feature_selection.data_mngt import read_data, split_data
from feature_selection.data_preprocessing import imbalance_check, label_encoding, scale_data
from feature_selection.models import dtree, rforest, xgboost, perm_knn, chi_2, mutual_inf, categorical_corr, unc_coeff, anova, log_reg, svm
from feature_selection.utils.main_utils import princ_comp_anal, merge_plots, make_timestamp_dir, compare_metrics
from feature_selection.intrinsic.tree_based import TreeBasedModels

from feature_selection.constants.datasets import DATASETS
from feature_selection.constants import *

## follow PEP8 standards 
# Class names must be camelcase (Ex: DataManagement)
# function and variable names must be lowercase with words separated by underscore (Ex: read_data, file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for getting dataset from command-line')
    parser.add_argument("-d", "--dataset", default="5. PHQ9_GAD7")

    # parsing command line arguments
    args = parser.parse_args()
    
    datasets_dir = DATASETS_DIR
    dataset_name = args.dataset
    csv_file_name = DATASETS[dataset_name]

    csv_file_path = os.path.join(datasets_dir, dataset_name, csv_file_name)
    df = read_data(csv_file_path)

    # Make a time stamp directory where storing the plots
    cur_time_stamp = make_timestamp_dir(dataset_name)

    # checking for class imbalance
    print("Imbalanced classes:",imbalance_check(df))
    logging.info(f"Imbalanced classes:{imbalance_check(df)}")

    # splitting the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, threshold=SPLIT_THRESHOLD)

    # scaling data using standard scaler by default(use scaler="minmax" for minmax scaling)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
    
    # encoding labels into 0 and 1
    y_train_encoded = label_encoding(y_train)
    y_test_encoded = label_encoding(y_test)
    assert y_train.value_counts().loc['H'] == y_train_encoded.value_counts().loc[1]
    
    print_features = PRINT_FEATURES
    n_features_to_select = NUM_FEATURES_TO_SELECT

    tree_methods = TreeBasedModels(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, cur_time_stamp, print_features, n_features_to_select)

    # model training for feature selection
    dtree_metrics = tree_methods.model(DECISION_TREE)
    print("end of decision tree".center(50,"*"))

    rforest_metrics = tree_methods.model(RANDOM_FOREST)
    print("end of random forest".center(50,'*'))

    xgboost_metrics = tree_methods.model(XGBOOST)
    print("end of xgboost".center(50,'*'))

    
    logreg_metrics = log_reg(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, cur_time_stamp, print_features)
    print("end of logistic regression".center(50,'*'))

    svm_metrics = svm(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, cur_time_stamp, print_features)
    print("end of support vector machine".center(50,'*'))

    plot_perm = perm_knn(X_train_scaled, y_train_encoded, cur_time_stamp, n_features_to_select, print_features)
    print("end of permutation importances with knn".center(50,'*'))

    selected_features_chi2 = chi_2(X_train, y_train_encoded, X_test, cur_time_stamp, n_features_to_select, print_features)
    print("end of chi2 feature selection".center(50,'*'))

    selected_features_mutualinf = mutual_inf(X_train, y_train_encoded, X_test, cur_time_stamp, n_features_to_select, print_features)
    print("end of mutual information feature selection".center(50,'*'))

    selected_features_anova = anova(X_train, y_train_encoded, X_test, cur_time_stamp, n_features_to_select, print_features)
    print("end of anova feature selection".center(50,'*'))

    categorical_corr(df, cur_time_stamp)
    print("end of categorical correlation study".center(50,'*'))

    unc_coeff(df, cur_time_stamp)
    print("end of uncertainty coefficients study".center(50,'*'))

    print("Start reduction of dimensionality using PCA".center(50,'*'))
    X_pca = princ_comp_anal(X_train_scaled, cur_time_stamp)

    # Merge all different plots in one figure and save it
    merge_plots(cur_time_stamp , "combined.png")

    # Reduce number of features
    X_train_reduced = X_train_scaled.loc[:,selected_features_chi2]
    X_test_reduced = X_test_scaled.loc[:,selected_features_chi2]

    # Try the models with the reduced features
    dtree_metrics_red = dtree(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, cur_time_stamp, print_features, plot = False)
    print("end of decision tree".center(50,"*"))

    rforest_metrics_red = rforest(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, cur_time_stamp, print_features, plot = False)
    print("end of random forest".center(50,'*'))

    xgboost_metrics_red = xgboost(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, cur_time_stamp, print_features, plot = False)
    print("end of xgboost".center(50,'*'))

    logreg_metrics_red = log_reg(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, cur_time_stamp, print_features, plot = False)
    print("end of xgboost".center(50,'*'))

    svm_metrics_red = svm(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, cur_time_stamp, print_features, plot = False)
    print("end of xgboost".center(50,'*'))


    # Compare metrics
    compare_metrics(dtree_metrics, dtree_metrics_red, "Decision tree")
    compare_metrics(rforest_metrics, rforest_metrics_red, "Random Forest")
    compare_metrics(xgboost_metrics, xgboost_metrics_red, "XGBoost")
    compare_metrics(logreg_metrics, logreg_metrics_red, "Logistic regression")
    compare_metrics(svm_metrics, svm_metrics_red, "Support vector machine")

