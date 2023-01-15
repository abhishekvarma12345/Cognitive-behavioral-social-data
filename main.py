import os
from feature_selection.data_mngt import read_data, split_data
from feature_selection.data_preprocessing import imbalance_check, label_encoding, scale_data, smote_tomek
from feature_selection.models import dtree, rforest, xgb, perm_knn, chi_2, mutual_inf, categorical_corr, unc_coeff, anova, log_reg, svm
from feature_selection.utils import pca, how_many_common, errorbars, plot_stability_map, new_features_training, pca_comparison
from feature_selection.utils import bar_plot, model_accuracy_comparison, heatmap, heatmap_jaccard, print_datasets
from feature_selection.utils import make_timestamp_dir, mean_change_accuracy, models_trn
import numpy as np

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

    print_datasets(folders_and_files)
    datasets_dir = os.path.join(os.getcwd(), 'Datasets')
    df_n = int(input("\nChoose the dataset from the list above (6 and 10 take hours to run): \n"))
    percentage_features_to_select = float(input("Enter fraction of features to select, or press enter for default (0.2): \n") or "0.2")
    folder_name = folders_and_files[df_n][0]
    filename = folders_and_files[df_n][1]
    file_path = os.path.join(datasets_dir, folder_name, filename)
    df = read_data(file_path)


    # Make a time stamp directory where storing the plots
    mydir = make_timestamp_dir(folder_name)

    # brief exploratory data analysis
    bar_plot(df, mydir, f'bar_plot_{df_n}.pdf')
    errorbars(df, mydir, f"mean_and_std_{df_n}.pdf")


    # splitting the data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df, threshold=0.8)

    # Balance dataset number 10
    if filename == folders_and_files[10][1] :
        X_train, y_train = smote_tomek(X_train, y_train)

    # Checking for class imbalance
    print("Are classes imbalanced? ", imbalance_check(df))

    # Scaling data using standard scaler by default(use scaler="minmax" for minmax scaling)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test, scaler = 'min_max')
    X_train_scaled_H = X_train.loc[y_train == 'H'] # To try the PCA with honest answers

    # Encoding labels into 0 and 1
    y_train_encoded = label_encoding(y_train)
    y_test_encoded = label_encoding(y_test)
    assert y_train.value_counts().loc['H'] == y_train_encoded.value_counts().loc[1]
    
    # Set some parameters for functions that will be used later
    print_features = False
    n_features_to_select = int(percentage_features_to_select * (len(df.columns)-1))
    if n_features_to_select == 1: n_features_to_select = 2 # Force to have at least two selected features
    n_features_list = list(np.arange(1,len(df.columns),1)) # list with as many integers as features in the daaset
    models = [dtree, rforest, xgb, log_reg, svm] # ML models for classification
    selection_methods = [chi_2, mutual_inf, anova, perm_knn, pca] # Model independent feature selection methods

    
    #############################################################################################################################################

    # Classify and find importance of all features with ML models
    metrics_all_fts, importances_all_fts, model_features_list = models_trn(X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, mydir, models, n_features_to_select, df_n, False, True)

    # Study the correlation of features
    categorical_corr(df, mydir, df_n)
    unc_coeff(df, mydir, df_n)


    # Select specific number of features with different feature selection methods
    selected_features_perm = perm_knn(X_train_scaled, y_train_encoded, X_test, mydir, n_features_to_select,df_n, print_features)
    selected_features_chi2 = chi_2(X_train, y_train_encoded, X_test, mydir, n_features_to_select,df_n, print_features)
    selected_features_mutualinf = mutual_inf(X_train, y_train_encoded, X_test, mydir, n_features_to_select, df_n, print_features)
    selected_features_anova = anova(X_train, y_train_encoded, X_test, mydir, n_features_to_select, df_n, print_features)
    selected_features_pca = pca(X_train_scaled, mydir, n_features_to_select, df_n)
    selected_features_list = [selected_features_chi2, selected_features_mutualinf, selected_features_anova, selected_features_perm, selected_features_pca]
    
    # Print selected features with each method
    print(f"Selected pca {selected_features_pca}")
    print(f"Selected chi2 {selected_features_chi2}")
    print(f"Selected mut_info {selected_features_mutualinf}")
    print(f"Selected anova {selected_features_anova}")
    print(f"Selected permutation {selected_features_perm}")


#######################################################################################################################################################################################
# Study of the stability of the chosen features 
#######################################################################################################################################################################################
    
    # See stability with all features
    plot_stability_map(importances_all_fts, mydir, df.columns[:-1] , f"stability_all_features_{df_n}.pdf")

    # Train model with selected features from each method. COmpare, get importances.
    importances_red_fts = new_features_training(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, mydir, df_n, models, n_features_to_select, selected_features_list, metrics_all_fts, selection_methods)

    # See stability after choosing 20% features with chi_2
    plot_stability_map(importances_red_fts[0], mydir, selected_features_chi2, f'stability_sel_features_chi2_{df_n}.pdf')
    plot_stability_map(importances_red_fts[1], mydir, selected_features_mutualinf, f'stability_sel_features_mutinf_{df_n}.pdf')
    plot_stability_map(importances_red_fts[2], mydir, selected_features_anova, f'stability_sel_features_anova_{df_n}.pdf')
    plot_stability_map(importances_red_fts[3], mydir, selected_features_perm, f'stability_sel_features_perm_{df_n}.pdf')
    plot_stability_map(importances_red_fts[4], mydir, selected_features_pca, f'stability_sel_features_pca_{df_n}.pdf')

    
    # print the common features chosen by all selection methods
    how_many_common(selected_features_list, mydir, df_n)

    # Print Jaccard similatiry heatmap for all ML+methods and only for mehotds
    heatmap_jaccard(selected_features_list + model_features_list, mydir, models_list=['DT','RF','XGB', 'LR', 'SVM'], filename = f'Jaccard_all_{df_n}.pdf')
    heatmap_jaccard(selected_features_list, mydir, filename = f"Jaccard_selectors_{df_n}.pdf")

#######################################################################################################################################################################################
# Study of the stability of the accuracy
#######################################################################################################################################################################################


    # # Plot change in accuracy after running the models with the chosen features
    heatmap(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
            mydir, metrics_all_fts, selection_methods, models,
            selected_features_list, n_features_to_select, df_n)

    # for each selection method plot the change in accuracy that each model presents 
    # when different number of features are chosen 
    for selector in selection_methods:
        model_accuracy_comparison(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
                                  mydir, metrics_all_fts, selector, models, n_features_list,
                                  n_features_to_select, df_n)
    
    # Compare the different feature selectors. 
    # Plot mean change in accuracy with respect to number of features chosen. 
    # The mean is done on the change of accuracy of each model.)
    mean_change_accuracy(X_train_scaled, X_test_scaled, y_train_encoded, 
            y_test_encoded, mydir, metrics_all_fts, selection_methods, 
            models, n_features_list, n_features_to_select, df_n)

    # Compare both pca methods (only honest and honest+dishonest)
    pca_comparison(X_train_scaled, X_train_scaled_H, X_test_scaled, y_train_encoded, 
                         y_test_encoded, mydir, metrics_all_fts, models, 
                         n_features_list, n_features_to_select, df_n)




