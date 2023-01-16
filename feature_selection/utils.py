import os, datetime
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
import scipy.stats as ss
import numpy as np
from dython.nominal import conditional_entropy
from dython.nominal import Counter
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import metrics
import pandas as pd

def make_timestamp_dir(folder_name):
    """
    Function to create a timestamped directory

    Parameters:
    - folder_name : str, name of the folder

    Returns:
    - mydir : str, path of the newly created timestamped folder
    """

    mydir = os.path.join(os.getcwd(), 'feature_selection', 'artifacts', folder_name,
                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(mydir, exist_ok=True)

    return mydir

def save_plot(columns, feature_importances, filename, dir):
    """
    Function to save a bar plot of feature importances

    Parameters:
    - columns : list, feature names
    - feature_importances : list, values of feature importances
    - filename : str, name of the file to save the plot
    - dir : str, directory to save the plot

    Returns:
    - fig : matplotlib figure object
    """
    fig = plt.figure(figsize=(16,12))
    plt.bar(columns, feature_importances)
    plt.xlabel("Features", fontsize = 25)
    plt.ylabel("Feature importance", fontsize = 25)
    plt.xticks(fontsize = 20, rotation = 45)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, filename), dpi = 500)
    return fig

def select_features(X_train, y_train, X_test, score_f):
    """
    Function to perform feature selection using a given scoring function

    Parameters:
    - X_train : pd.DataFrame, training set features
    - y_train : pd.DataFrame, training set target
    - X_test : pd.DataFrame, testing set features
    - score_f : scoring function, function to evaluate the importance of features

    Returns:
    - X_train_fs : transformed training set features
    - X_test_fs : transformed testing set features
    - fs : fitted feature selector object
    """

    fs = SelectKBest(score_func=score_f, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def cramers_corrected_stat(confusion_matrix):
    """
    Function to compute Cramer's corrected V statistic for a 
    given confusion matrix. Cramers V statistic is calculated 
    for categorical-categorical association. Uses correction 
    from Bergsma and Wicher, Journal of the Korean Statistical 
    Society 42 (2013): 323-328.

    Parameters:
    - confusion_matrix : np.ndarray, confusion matrix for which 
      Cramer's corrected V statistic is to be computed

    Returns:
    - Cramer's corrected V statistic
    """

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def theils_u(x, y):
    """
    Function to compute Theil's U for two categorical variables

    Parameters:
    - x : list, first categorical variable
    - y : list, second categorical variable

    Returns:
    - Theil's U
    """
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def save_plot_sns(dat, filename, dir, x_name = "", y_name = "", title = ""):
    """
    Function to save a heatmap of a matrix with seaborn

    Parameters:
    - dat : 2D array, data to be plotted
    - filename : str, name of the file to save the plot
    - dir : str, directory to save the plot
    - x_name : str, name of x axis (default: "")
    - y_name : str, name of y axis (default: "")
    - title : str, title of the plot (default: "")

    Returns:
    - None
    """

    fig, ax = plt.subplots(figsize=(16,12))
    ax = sns.heatmap(dat, annot=True, ax=ax, annot_kws={"size" : 15}, fmt ='.1g')
    plt.title(title, fontsize = 30)
    plt.xlabel(x_name, fontsize = 25)
    plt.ylabel(y_name, fontsize = 25)
    plt.xticks(fontsize=20, rotation = 45)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, filename))


def pca(X, dir, n_features, df_n):
    """
    Function to perform PCA and return the selected feature names

    Parameters:
    - X : pd.DataFrame, input data
    - dir : str, directory to save plot
    - n_features : int, number of features to select
    - df_n (int): number of the dataset

    Returns:
    - list of selected feature names
    """

    p_c_a = PCA().fit(X)

    fig = plt.figure(figsize=(16,12))
    plt.plot(np.cumsum(p_c_a.explained_variance_ratio_), marker = '.')
    plt.xlabel('Number of components', fontsize = 25)
    plt.ylabel('Cumulative explained variance', fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"pca_{df_n}.pdf"), dpi = 500)

    n_comp = (np.where(np.cumsum(p_c_a.explained_variance_ratio_) > 0.8))[0][0]
    if n_comp == 0: n_comp = 1 # force to choose at least one component

    principal = PCA(n_components=n_comp)
    principal.fit(X)
    X_pca = principal.transform(X)

    # correlation matrix
    X_pca = pd.DataFrame(X_pca)
    cols = X.columns
    cols_pca = X_pca.columns
    corrM = np.zeros((len(cols), len(cols_pca)))

    for i in range(len(cols)):
        for j in range(len(cols_pca)):
            corr = X.iloc[:,i].corr(X_pca.iloc[:,j])
            corrM[i][j] = corr

    corrM = np.abs(corrM)
    max_row = np.amax(corrM, axis = 1)
    max = np.argsort(max_row)[-n_features:]

    return list(cols[max])


def get_metrics(y_test, y_pred):
    """
    Function to calculate evaluation metrics for a classification model

    Parameters:
    - y_test : array-like, true target values
    - y_pred : array-like, predicted target values

    Returns:
    - metrics_dict : dict, containing accuracy, roc_auc, f1, precision, recall scores
    """
    metrics_dict = {"Accuracy" : metrics.accuracy_score(y_test, y_pred),
                    "Roc_auc" : metrics.roc_auc_score(y_test, y_pred),
                    "F1" : metrics.f1_score(y_test, y_pred),
                    "Precision" : metrics.precision_score(y_test, y_pred),
                    "Recall" : metrics.recall_score(y_test, y_pred)}

    return metrics_dict


def compare_metrics(dict_full, dict_selected, model_name, method_name = None, show_metrics = True):
    """
    Function to compare the evaluation metrics of two classification models

    Parameters:
    - dict_full : dict, evaluation metrics of the model with all features
    - dict_selected : dict, evaluation metrics of the model with selected features
    - model_name : str, name of the model
    - show_metrics : bool, whether to print the metrics or not (default: True)

    Returns:
    - Change in accuracy
    """
    new_dict = {}

    for key in dict_full.keys():
        new_dict[key] = [round(dict_full[key], 2), round(dict_selected[key], 2), round(100*(dict_selected[key]-dict_full[key])/dict_full[key], 2)] 

    df = pd.DataFrame.from_dict(new_dict)
    df["Features"] = ["All", "Selected", "Change (%)"]

    if show_metrics:
        print(f"\n Comparison for {model_name}, features selected with {method_name}")
        print(df.set_index("Features")) 

    return df["Accuracy"].iloc[2]




def bar_plot(df, dir, filename):
    """
    Function to create a bar chart of the mean value of a given dataframe by group

    Parameters:
    - df : pd.DataFrame, containing the data to be plotted
    - dir : str, directory to save the plot
    - filename : str, name of the file to save the plot

    Returns:
    - None
    """
    
    df_grouped = df.groupby(by=['CONDITION'], as_index = False).mean()

    df_to_plot = pd.DataFrame({
    'Columns': df_grouped.columns[1:],
    'H': df_grouped[df_grouped['CONDITION'] == 'H'].values.flatten().tolist()[1:],
    'D': df_grouped[df_grouped['CONDITION'] == 'D'].values.flatten().tolist()[1:]
    })

    fig, ax = plt.subplots(figsize=(16,12))  
    ax = df_to_plot.plot.bar(x = 'Columns' , y = ['H','D'] , rot=45, xlabel = "Features")
    plt.savefig(os.path.join(dir, filename), dpi = 500)



def how_many_common(selected_feat_list, mydir, df_n):
    """
    Function to determine the common features selected 
    by different feature selection methods.

    Parameters:
    - selected_feat_list : list of lists, containing the 
      selected feature names by different feature 
      selection methods
    - mydir : str, directory to save the plot
    - df_n (int): number of the dataset

    Returns:
    - None
    """

    l = selected_feat_list[0]

    for feat_list in selected_feat_list[1:]:
        l = list(set(l).intersection(feat_list))

    print(f"Out of {len(selected_feat_list[0])} selected features, the following {len(l)} are common to all methods: {l}")

    # Plot bar plot of frequencies
    toghether = [elem for lst in selected_feat_list for elem in lst] 
    counts = Counter(toghether)
    df = pd.DataFrame.from_dict(counts, orient='index')

    fig, ax = plt.subplots(figsize=(16,12)) 
    df.plot(kind='bar', rot=45, legend = False)
    plt.xlabel("Selected features", fontsize = 25)
    plt.ylabel("Frequency", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, f"common_features_{df_n}.pdf"), dpi = 500)


def models_trn(X_train, y_train, X_test, y_test, mydir, models, n_feat_to_select, df_n, print_fts, plot_or_not):
    """
    Function to train multiple models and return the evaluation metrics, feature importances and selected features

    Parameters:
    - X_train : pd.DataFrame, training set features
    - y_train : pd.DataFrame, training set target
    - X_test : pd.DataFrame, testing set features
    - y_test : pd.DataFrame, testing set target
    - mydir : str, directory to save plots
    - models : list of functions, list of models to be trained
    - n_feat_to_select : int, number of features to select
    - print_fts : bool, whether to print the feature importances or not (default: True)
    - plot_or_not : bool, whether to save the feature importances plot or not (default: True)
    - df_n (int): number of the dataset

    Returns:
    - metrics_list : list of dictionaries, containing evaluation metrics for each model
    - importance_list : list of arrays, containing feature importances for each model
    - models_list : list of lists, containing the selected feature names for each model
    """

    dtree_metrics, importance_dtree, feat_lst_dt  = models[0](X_train, y_train, X_test, 
            y_test,n_feat_to_select, mydir, df_n, print_features = print_fts, plot = plot_or_not)
    rforest_metrics, importance_forest, feat_lst_rf = models[1](X_train, y_train, X_test, 
            y_test,n_feat_to_select, mydir, df_n, print_features = print_fts, plot = plot_or_not)
    xgboost_metrics, importance_xgb, feat_lst_xgb = models[2](X_train, y_train, X_test, 
            y_test,n_feat_to_select, mydir, df_n, print_features = print_fts, plot = plot_or_not)
    logreg_metrics, importance_logreg, feat_lst_lr = models[3](X_train, y_train, X_test, 
            y_test,n_feat_to_select, mydir, df_n, print_features = print_fts, plot = plot_or_not)
    svm_metrics, importance_svm, feat_lst_svm = models[4](X_train, y_train, X_test, 
            y_test,n_feat_to_select, mydir, df_n, print_features = print_fts, plot = plot_or_not)

    importance_list = [importance_dtree, importance_forest, importance_xgb, importance_logreg, importance_svm]
    metrics_list = [dtree_metrics, rforest_metrics, xgboost_metrics, logreg_metrics, svm_metrics]
    models_list = [feat_lst_dt, feat_lst_rf, feat_lst_xgb, feat_lst_lr, feat_lst_svm]

    return metrics_list, importance_list, models_list



def model_accuracy_comparison(X_train_scaled, X_test_scaled, y_train_encoded,
                              y_test_encoded, mydir, metrics_all_fts, selector,
                              models, n_features_list, n_features_to_select, df_n):
    """
    This function compares the accuracy of various models using different number of features selected using a given feature selection method
    
    Parameters:
    - X_train_scaled (DataFrame): The scaled training data
    - X_test_scaled (DataFrame): The scaled test data
    - y_train_encoded (Series): The encoded training labels
    - y_test_encoded (Series): The encoded test labels
    - mydir (str): File path to save plots
    - metrics_all_fts (list of dicts): List of dictionaries containing metrics of all models on the original data
    - selector (function): The feature selection function to use
    - models (list of models): List of models to train
    - n_features_list (list of ints): List of number of features to select
    - n_features_to_select (int): Number of features to select for final model
    - df_n (int): number of the dataset
    """ 
    acc_tree, acc_for, acc_xgb, acc_lr, acc_svm = [], [], [], [], []

    for i in n_features_list:

        if str(selector).split(' ')[1] == 'pca':
            selected_features = selector(X_train_scaled, mydir, i, df_n)
        else:
            selected_features = selector(X_train_scaled, y_train_encoded, X_test_scaled, mydir, i, df_n, print_features = False)
        
        X_train_reduced = X_train_scaled.loc[:,selected_features]
        X_test_reduced = X_test_scaled.loc[:,selected_features]

        metrics_red, *_ = models_trn(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, 
                                     mydir, models,n_features_to_select, df_n, False, False)

        acc_tree.append(compare_metrics(metrics_all_fts[0], metrics_red[0], "Decision tree", show_metrics = False))
        acc_for.append(compare_metrics(metrics_all_fts[1], metrics_red[1], "Random Forest", show_metrics = False))
        acc_xgb.append(compare_metrics(metrics_all_fts[2], metrics_red[2], "XGBoost", show_metrics = False))
        acc_lr.append(compare_metrics(metrics_all_fts[3], metrics_red[3], "Logistic regression", show_metrics = False))
        acc_svm.append(compare_metrics(metrics_all_fts[4], metrics_red[4], "Support vector machine", show_metrics = False))


    name_plot = str(selector).split(' ')[1] + f"_behavior_number_features_{df_n}.pdf"
    fig = plt.figure(figsize=(16,12))
    plt.plot(n_features_list, acc_tree, label = "Decision tree", marker='.')
    plt.plot(n_features_list, acc_for, label = "Random forest", marker='.')
    plt.plot(n_features_list, acc_xgb, label = "XGBoost", marker='.')
    plt.plot(n_features_list, acc_lr, label = "Logistic regression", marker='.')
    plt.plot(n_features_list, acc_svm, label = "SVM", marker='.')
    plt.axvline(x = n_features_to_select, color = 'k', linestyle='dashdot', alpha = 0.5)
    plt.xlabel("Number of selected features", fontsize = 25)
    plt.ylabel("Change in accuracy (%)", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    # plt.title(f"{str(selector).split(' ')[1]}", fontsize=25)
    plt.legend(prop={'size': 18})
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, name_plot), dpi = 500)


def heatmap(X_train_scaled, X_test_scaled, y_train_encoded, 
            y_test_encoded, mydir, metrics_all_fts, 
            selection_methods, models, selected_features_list, 
            n_feat_to_select, df_n):

    """
    This function plots a heatmap of the change in accuracy of various models
    using different feature selection methods.
    
    Parameters:
    - models (list of models): list of models to train
    - selection_methods (list of functions): list of feature selection methods
    - matrix (2D array): matrix of change in accuracy of different models and feature selection methods
    - mydir (str): File path to save the heatmap
    - df_n (int): number of the dataset
    """
    matrix = np.zeros((len(selection_methods), len(models)))

    for i in range(len(selected_features_list)):

        X_train_reduced = X_train_scaled.loc[:, selected_features_list[i]]
        X_test_reduced = X_test_scaled.loc[:, selected_features_list[i]]

        metrics_red, *_ = models_trn(X_train_reduced, y_train_encoded, X_test_reduced, 
                            y_test_encoded, mydir, models,n_feat_to_select, df_n, False, False)

        matrix[i][0] = compare_metrics(metrics_all_fts[0], metrics_red[0], "Decision tree", show_metrics = False)
        matrix[i][1]= compare_metrics(metrics_all_fts[1], metrics_red[1], "Random Forest", show_metrics = False)
        matrix[i][2]= compare_metrics(metrics_all_fts[2], metrics_red[2], "XGBoost", show_metrics = False)
        matrix[i][3]= compare_metrics(metrics_all_fts[3], metrics_red[3], "Logistic regression", show_metrics = False)
        matrix[i][4]= compare_metrics(metrics_all_fts[4], metrics_red[4], "Support vector machine", show_metrics = False)

    # Create dataset with the data to easily make the heatmap afterwards
    labels_selectors = [str(i).split(' ')[1] for i in selection_methods]
    labels_models = [str(i).split(' ')[1] for i in models]
    dat = pd.DataFrame(matrix, index=labels_selectors, columns=labels_models)
    # plot the heatmap
    save_plot_sns(dat, f"heatmap_accuracy_{df_n}.pdf", mydir, x_name = "Models", y_name = "Selection methods", title = "")



def mean_change_accuracy(X_train_scaled, X_test_scaled, y_train_encoded, 
                         y_test_encoded, mydir, metrics_all_fts, selection_methods, models, 
                         n_features_list, n_feat_to_select, df_n):
    """
    This function plots the mean change in accuracy of various models using different feature selection methods.
    
    Parameters:
    - X_train_scaled (DataFrame): The scaled training data
    - X_test_scaled (DataFrame): The scaled test data
    - y_train_encoded (Series): The encoded training labels
    - y_test_encoded (Series): The encoded test labels
    - mydir (str): File path to save the plot
    - metrics_all_fts (list of dicts): List of dictionaries containing metrics of all models on the original data
    - selection_methods (list of functions): List of feature selection methods
    - models (list of models): List of models to train
    - n_features_list (list of ints): List of number of features to select
    - n_feat_to_select (int): Number of features to select for final model
    - df_n (int): number of the dataset
    """

    my_list_3 = []
    for selector in selection_methods:

        my_list_2 = []
        # acc_chi2, acc_mutinf, acc_anova, acc_perm = [], [], [], []

        for i in n_features_list:   

            if str(selector).split(' ')[1] == 'pca':
                selected_features = selector(X_train_scaled, mydir, i, df_n)
            else:
                selected_features = selector(X_train_scaled, y_train_encoded, X_test_scaled, 
                                            mydir, i, df_n, print_features = False)

            X_train_reduced = X_train_scaled.loc[:, selected_features]
            X_test_reduced = X_test_scaled.loc[:, selected_features]

            metrics_red, *_ = models_trn(X_train_reduced, y_train_encoded, X_test_reduced, 
                                    y_test_encoded, mydir, models, n_feat_to_select,df_n, False, False)

            my_list = [
            compare_metrics(metrics_all_fts[0], metrics_red[0], "Decision tree", show_metrics = False),
            compare_metrics(metrics_all_fts[1], metrics_red[1], "Random Forest", show_metrics = False),
            compare_metrics(metrics_all_fts[2], metrics_red[2], "XGBoost", show_metrics = False),
            compare_metrics(metrics_all_fts[3], metrics_red[3], "Logistic regression", show_metrics = False),
            compare_metrics(metrics_all_fts[4], metrics_red[4], "Support vector machine", show_metrics = False)
            ]

            avg = sum(my_list)/len(my_list)
            my_list_2.append(avg)

        my_list_3.append(my_list_2)


    fig = plt.figure(figsize=(16,12))
    plt.plot(n_features_list, my_list_3[0], label = "Chi2", marker='.')
    plt.plot(n_features_list, my_list_3[1], label = "Mutual information", marker='.')
    plt.plot(n_features_list, my_list_3[2], label = "Anova", marker='.')
    plt.plot(n_features_list, my_list_3[3], label = "Permutation importance", marker='.')
    plt.plot(n_features_list, my_list_3[4], label = "PCA", marker='.')
    plt.axvline(x = n_feat_to_select, color = 'k', linestyle='dashdot', alpha = 0.5)
    plt.xlabel("Number of selected features", fontsize = 25)
    plt.ylabel("Mean change in accuracy (%)", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(prop={'size': 18})
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, f"mean_change_in_accuracy_{df_n}.pdf"), dpi = 500)



def errorbars(df, mydir, filename):
    """
    This function creates a errorbar plot of mean choice by features 
    grouped by a given condition ('CONDITION' column in the dataframe)
    
    Parameters:
        df (dataframe): Dataframe containing the data to be plotted
        mydir (str): File path to save the plot
        filename (str): File name to save the plot
    """
    grouped_df_mean = df.groupby(by="CONDITION").mean()
    grouped_df_std = df.groupby(by="CONDITION").std()

    plt.figure(figsize=(16,12)) 
    plt.errorbar(grouped_df_mean.columns, grouped_df_mean.loc['H'], grouped_df_std.loc['H'], capsize=6, label= "H")
    plt.errorbar(grouped_df_mean.columns, grouped_df_mean.loc['D'], grouped_df_std.loc['D'], capsize=4, label = "D")
    plt.xticks(rotation=45, ha='right')
    plt.legend(prop={'size': 18})
    plt.xlabel("Features", fontsize = 25)
    plt.ylabel("Mean choice", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    # plt.title(f"{dataset_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, filename), dpi = 500)



def normalize_importance(importance):
    """
    This function takes a list of importance values and normalizes them between 0 and 1.

    Input:
    - importance: list of numbers

    Output:
    - norm: list of normalized numbers
    """
    minim = min(importance)
    maxim = max(importance)
    norm = [(float(i)-minim)/(maxim-minim) for i in importance]
    return norm


def plot_stability_map(lst, mydir, columns, name):
    """
    This function plots a stability map of feature importances for different models.

    Input:
    - lst: list of lists of feature importances for different models
    - mydir: directory to save the plot
    - columns: list of feature names
    - name: name of the plot file

    Output:
    - None
    """
    Index = ['d_tree', 'r_forest', 'xgb', 'log_reg', 'svm']
    lst_norm = [normalize_importance(ls) for ls in lst]
    dataf = pd.DataFrame(lst_norm, columns = columns, index=Index, dtype = float)

    save_plot_sns(dataf, name, mydir, "Features", "Models","" )



def jaccard_set(list1, list2):
    """
    This function calculates the Jaccard similarity between two sets of items.

    Input:
    - list1: list of items
    - list2: list of items

    Output:
    - jaccard_similarity: float value representing the similarity between the two input sets
    """

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return intersection / union



def heatmap_jaccard(selected_feature_list, mydir, models_list = [], filename = None):
    """
    This function creates a heatmap of Jaccard similarities between different feature selection methods.

    Input:
    - selected_feature_list: list of lists of selected features for different feature selection methods
    - mydir: directory to save the plot
    - models_list: list of strings, names of machine learning models (optional)
    - filename: name of the plot file (optional)

    Output:
    - None
    """
    size = len(selected_feature_list)
    matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            matrix[i][j] = jaccard_set(selected_feature_list[i], selected_feature_list[j])

    columns = ['Chi_2','M_inf.','Anova', 'Perm', 'PCA'] + models_list
    df = pd.DataFrame(matrix, columns = columns, index=columns)
    save_plot_sns(df, filename, mydir)



def new_features_training(X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, mydir, df_n, models, 
                          n_features_to_select, selected_features, metrics_all_fts, selection_methods):
    """
    Trains models with a reduced set of features and compares their performance to models trained with all features.

    Parameters:
    - X_train_scaled (DataFrame): Scaled training dataset
    - X_test_scaled (DataFrame): Scaled testing dataset
    - y_train_encoded (DataFrame): Encoded training labels
    - y_test_encoded (DataFrame): Encoded testing labels
    - mydir (str): directory path for the results
    - df_n (int): number of the dataset
    - models (List[Estimator]): List of models to train
    - n_features_to_select (int): Number of features to select
    - selected_features (List[List[str]]): List of lists of selected features for each feature selection method
    - metrics_all_fts (List[Dict]): List of dictionaries containing metrics of the models trained with all features
    - selection_methods (List[str]): List of feature selection method names

    Returns:
    - importances_red_features (List[Dict]): List of dictionaries containing feature importances of the models 
      trained with reduced features
    """

    importances_red_features = []

    for i in range(len(selected_features)):
        # Create new X_train and X_test variables with the chosen features
        X_train_reduced = X_train_scaled.loc[:,selected_features[i]]
        X_test_reduced = X_test_scaled.loc[:,selected_features[i]] 

        metrics_red_fts, importances_red_fts, _ = models_trn(X_train_reduced, y_train_encoded, 
                                                             X_test_reduced, y_test_encoded, mydir, 
                                                             models, n_features_to_select, df_n, False, False)

        importances_red_features.append(importances_red_fts)

        # Compare metrics (acc, f1, recall...) of ML models with all features and with the selected number of features, for all feature selection methods.
        for j in range(len(models)):
            compare_metrics(metrics_all_fts[j], metrics_red_fts[j], model_name = str(models[j]).split(' ')[1], 
                            method_name = str(selection_methods[j]).split(' ')[1])

    return importances_red_features



def print_datasets(folders_and_files):
    """
    Prints a list of datasets in a given list of folders and files.

    Parameters:
    - folders_and_files (list): a list of tuples containing a string of the 
      folder/file name and a boolean value indicating if it's a folder or a file

    Returns:
    - None
    """
    print("List of datasets: \n")
    for i in range(len(folders_and_files)):
        print(i,folders_and_files[i][0].split()[1])




def pca_comparison(X_train_scaled, X_train_scaled_H, X_test_scaled, y_train_encoded, 
                         y_test_encoded, mydir, metrics_all_fts, models, 
                         n_features_list, n_feat_to_select, df_n):

    """
    Compare the performance of models trained with features selected by PCA with 
    the performance of models trained with all features.

    Parameters:
    - X_train_scaled (DataFrame): Scaled training dataset
    - X_train_scaled_H (DataFrame): Scaled training dataset with only honest samples
    - X_test_scaled (DataFrame): Scaled testing dataset
    - y_train_encoded (DataFrame): Encoded training labels
    - y_test_encoded (DataFrame): Encoded testing labels
    - mydir (str): directory path for the results
    - metrics_all_fts (List[Dict]): List of dictionaries containing metrics of 
      the models trained with all features
    - models (List[Estimator]): List of models to train
    - n_features_list (List[int]): List of the number of features to select
    - n_feat_to_select (int): Number of features to select
    - df_n (int): number of the dataset

    Returns:
    - None
    """
    my_list_3 = []
    for X_tr in [X_train_scaled, X_train_scaled_H]:

        my_list_2 = []

        for i in n_features_list:   

            selected_features = pca(X_tr, mydir, i, df_n)

            X_train_reduced = X_train_scaled.loc[:, selected_features]
            X_test_reduced = X_test_scaled.loc[:, selected_features]

            metrics_red, *_ = models_trn(X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, models, n_feat_to_select,df_n, False, False)

            my_list = [
            compare_metrics(metrics_all_fts[0], metrics_red[0], "Decision tree", show_metrics = False),
            compare_metrics(metrics_all_fts[1], metrics_red[1], "Random Forest", show_metrics = False),
            compare_metrics(metrics_all_fts[2], metrics_red[2], "XGBoost", show_metrics = False),
            compare_metrics(metrics_all_fts[3], metrics_red[3], "Logistic regression", show_metrics = False),
            compare_metrics(metrics_all_fts[4], metrics_red[4], "Support vector machine", show_metrics = False)
            ]

            avg = sum(my_list)/len(my_list)
            my_list_2.append(avg)

        my_list_3.append(my_list_2)


    fig = plt.figure(figsize=(16,12))
    plt.plot(n_features_list, my_list_3[0], label = "Honest + Dishonest", marker='.')
    plt.plot(n_features_list, my_list_3[1], label = "Honest", marker='.')
    plt.axvline(x = n_feat_to_select, color = 'k', linestyle='dashdot', alpha = 0.5)
    plt.xlabel("Number of selected features", fontsize = 25)
    plt.ylabel("Mean change in accuracy (%)", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(prop={'size': 18})
    plt.tight_layout()
    plt.savefig(os.path.join(mydir, f"pca_comparison_{df_n}.pdf"), dpi = 500)