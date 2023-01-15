from feature_selection.utils import save_plot,select_features , cramers_corrected_stat, theils_u, save_plot_sns, get_metrics

# intrinsic methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# dimensionality reduction methods
# instead of PCA we can use Fisher's Linear Discriminant
from sklearn.decomposition import PCA
from sklearn import metrics


# permutation importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance

# filter methods

# considering input datatype:ordinal, output datatype:categorical
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

# considering input datatype:numerical, output datatype:categorical
from sklearn.feature_selection import f_classif
# kendalls rank correlation method df.corr(method='kendall')


from sklearn.feature_selection import SelectFromModel, SelectKBest
import pandas as pd
import numpy as np
from itertools import combinations


def dtree(X, y, X_test, y_test, n_features_to_select, dir, df_n, print_features = True, plot = True):
    """
    This function fits a Decision Tree Classifier on the given train data, makes predictions 
    on the test data, and returns the evaluation metrics, feature importances and selected feature names.
    Parameters:
    - X: pd.DataFrame : Dataframe containing the input features for training
    - y: pd.Series : Series containing the target variable for training
    - X_test: pd.DataFrame : Dataframe containing the input features for testing
    - y_test: pd.Series : Series containing the target variable for testing
    - n_features_to_select : int : number of feature to select
    - dir : string : directory to save the plot
    - print_features : bool : whether to print feature importance scores
    - plot : bool : whether to plot feature importance scores
    - df_n (int): number of the dataset
    Returns:
    - metrics_dict : dict : evaluation metrics
    - importance : list : feature importance scores
    - selected_feat_names : list : names of selected features
    """
    # define the model
    model = DecisionTreeClassifier(random_state = 1)
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    metrics_dict = get_metrics(y_test, y_pred)

    # get importances
    importance = model.feature_importances_

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in zip(X.columns, importance):
            print('Feature', k, ':', round(v,2))

    if plot:
        save_plot(X.columns, importance, f'dtree_{df_n}.pdf', dir)

    return metrics_dict, importance, selected_feat_names


def rforest(X, y, X_test, y_test, n_features_to_select, dir, df_n, print_features = True, plot = True):
    """
    This function will train a Random Forest Classifier model on the input data, X and y and test it on 
    X_test and y_test. It will also return the feature importance scores, selected_feat_names and metrics_dict.

    Parameters:
    - X: input feature dataframe
    - y: input label dataframe
    - X_test: test feature dataframe
    - y_test: test label dataframe
    - n_features_to_select: number of features to select based on feature importance scores
    - dir: directory to save the feature importance plot
    - print_features: whether to print the feature importance scores or not
    - plot: whether to save the feature importance plot or not
    - df_n (int): number of the dataset

    Returns:
    - metrics_dict: A dictionary of evaluation metrics
    - importance: importance of features
    - selected_feat_names: names of features selected based on feature importance scores
    """
    model = RandomForestClassifier(random_state = 1)
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    metrics_dict = get_metrics(y_test, y_pred)

    # get importances
    importance = model.feature_importances_

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in zip(X.columns, importance):
            print('Feature', k, ':', round(v,2))

    if plot:
        save_plot(X.columns, importance, f'rforest_{df_n}.pdf', dir)

    return metrics_dict, importance, selected_feat_names


def xgb(X, y, X_test, y_test, n_features_to_select, dir, df_n, print_features = True, plot = True):
    """
    Function to train, predict and get feature importances of XGBoost model.

    Parameters:
    - X (DataFrame): Training dataframe.
    - y (Series): Target variable.
    - X_test (DataFrame): Test dataframe.
    - y_test (Series): Target variable of test dataframe.
    - n_features_to_select (int): number of features to select.
    - dir (str): directory where the plots will be saved.
    - print_features (bool): flag to decide whether to print feature importances or not.
    - plot (bool): flag to decide whether to save feature importance plot or not.
    - df_n (int): number of the dataset

    Returns:
    - tuple: metrics_dict, importance, selected_feat_names
    """
    # define the model
    model = XGBClassifier(random_state = 1)
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    metrics_dict = get_metrics(y_test, y_pred)

    importance = model.feature_importances_

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in zip(X.columns, importance):
            print('Feature', k, ':', round(v,2))

    if plot:
        save_plot(X.columns, importance, f'xgboost_{df_n}.pdf', dir)

    return metrics_dict, importance, selected_feat_names

def log_reg(X, y, X_test, y_test, n_features_to_select, dir, df_n, print_features = True, plot = True):
    """
    This function applies logistic regression to the input dataset and returns the metrics, importance 
    and selected features names

    Parameters:
    - X: array-like
    - y: array-like
    - X_test: array-like
    - y_test: array-like
    - n_features_to_select : int
    - dir : str
    - print_features : bool, default True
    - plot : bool, default True
    - df_n (int): number of the dataset

    Returns:
    - metrics_dict : dict
    - importance : array-like
    - selected_feat_names : list
    """

    # define the model
    model = LogisticRegression(random_state = 1)
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    metrics_dict = get_metrics(y_test, y_pred)

    importance = model.coef_[0]

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in zip(X.columns, importance):
            print('Feature', k, ':', round(v,2))

    # plot feature importance
    if plot:
        save_plot(X.columns, importance, f'log_reg_{df_n}.pdf', dir)

    return metrics_dict, importance, selected_feat_names


def svm(X, y, X_test, y_test, n_features_to_select, dir, df_n, print_features = True, plot = True):
    """
    This function defines and fits a Support Vector Machine model, predicts the test set, and returns
    the metrics, feature importances and the selected features.

    Parameters:
    - X : Dataframe
    The training feature set
    - y : Dataframe
    The training target set
    - X_test : Dataframe
    The test feature set
    - y_test : Dataframe
    The test target set
    - n_features_to_select : int
    The number of top features to select
    - dir : string
    The directory where the feature importance plot will be saved
    - print_features : boolean
    Whether to print feature importances or not
    - plot : boolean
    Whether to save the feature importance plot or not

    Returns:
    - metrics_dict : dict
    A dictionary containing the evaluation metrics of the model
    - importance : array
    A array containing the feature importances
    - selected_feat_names : list
    A list of the selected feature names
    - df_n (int): number of the dataset
    """
    # define the model
    model = SVC(kernel = "linear", random_state = 1)
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    metrics_dict = get_metrics(y_test, y_pred)

    importance = model.coef_[0]

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in zip(X.columns, importance):
            print('Feature', k, ':', round(v,2))

    # plot feature importance
    if plot:
        save_plot(X.columns, importance, f'svm_{df_n}.pdf', dir)

    return metrics_dict, importance, selected_feat_names


def perm_knn(X,y, X_test, dir, n_features_to_select, df_n, print_features = True):
    """
    Perform permutation feature importance for a KNeighborsClassifier model.

    Parameters:
    - X (pd.DataFrame) : Dataframe containing the training data
    - y (pd.Series) : Series containing the target variable
    - X_test (pd.DataFrame) : Dataframe containing the test data
    - dir (str) : directory to save the plot
    - n_features_to_select (int) : number of features to select
    - print_features (bool) : whether to print the feature importance scores or not
    - df_n (int): number of the dataset

    Returns:
    - List : List of selected feature names
    """
    # define the model
    model = KNeighborsClassifier()
    # fit the model
    model.fit(X, y)
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='accuracy')
    # get importance
    importance = results.importances_mean

    dict_name_score = dict(zip(X.columns, importance))
    max_scores = sorted(importance)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # summarize feature importance
    if print_features:
        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    save_plot(X.columns, importance, f'perm_knn_{df_n}.pdf', dir)
    return selected_feat_names 

 
def chi_2(X_train, y_train, X_test, dir, n_features_to_select, df_n, print_features = True):
    """
    Select the best features using chi-squared test

    Parameters:
    - X_train (DataFrame): Training data
    - y_train (Series): Target variable for training data
    - X_test (DataFrame): Test data
    - dir (str): directory where the plot will be saved
    - n_features_to_select (int): number of features to select
    - print_features (bool): whether to print the feature scores or not
    - df_n (int): number of the dataset

    Returns:
    - list : list of selected feature names
    """
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, chi2)

    dict_name_score = dict(zip(fs.get_feature_names_out(), fs.scores_))
    max_scores = sorted(fs.scores_)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in dict_name_score.items():
            print('Feature', k, ':', round(v,2))

    # plot the scores
    save_plot(X_train.columns, fs.scores_, f'chi_2_{df_n}.pdf', dir)
    return selected_feat_names


def mutual_inf(X_train, y_train, X_test, dir, n_features_to_select, df_n, print_features = True):
    """
    Calculate mutual information and select features using mutual_info_classif method

    Parameters:
    X_train (DataFrame): training set of features
    y_train (Series): training set of target
    X_test (DataFrame): test set of features
    dir (str): directory to save the plot of feature importance
    n_features_to_select (int): number of top important features to select
    print_features (bool): print the feature importance scores if set to True
    - df_n (int): number of the dataset

    Returns:
    list : list of selected feature names
    """
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif)
    

    dict_name_score = dict(zip(fs.get_feature_names_out(), fs.scores_))
    max_scores = sorted(fs.scores_)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in dict_name_score.items():
            print('Feature', k, ':', round(v,2))

    # plot the scores
    save_plot(X_train.columns, fs.scores_, f'mutual_inf_{df_n}.pdf', dir)
    return selected_feat_names


def categorical_corr(df, dir, df_n):
    """
    calculate correlation between categorical variables

    Parameters:
    - df (DataFrame): Dataframe containing the data
    - dir (str): path to save the plot
    - df_n (int): number of the dataset

    Returns:
    - None
    """
    cols = df.columns
    corrM = np.zeros((len(cols),len(cols)))
    for col1, col2 in combinations(cols, 2):
        idx1, idx2 = cols.get_loc(col1), cols.get_loc(col2)
        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    save_plot_sns(corr, f'categorical_correlation_{df_n}.pdf', dir)


def unc_coeff(df, dir, df_n):
    """
    Calculate Cramer's corrected V correlation coefficient between each pair of 
    categorical variables in a dataframe.

    Parameters:
    - df (DataFrame): Dataframe containing the data
    - dir (str): the directory where the plots will be saved
    - df_n (int): number of the dataset

    Returns:
    - None
    """

    cols = df.columns
    corrM = np.zeros((len(cols),len(cols)))
    for col1, col2 in combinations(cols, 2):
        idx1, idx2 = cols.get_loc(col1), cols.get_loc(col2)
        corrM[idx1, idx2] = theils_u(df[col1], df[col2])
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    save_plot_sns(corr, f'uncertainty_coefficients_{df_n}.pdf', dir)

def anova(X_train, y_train, X_test, dir, n_features_to_select, df_n, print_features = True):
    """
    Function to perform ANOVA feature selection and return the selected feature names.

    Parameters:
    - X_train : pd.DataFrame, training set features
    - y_train : pd.DataFrame, training set target
    - X_test : pd.DataFrame, testing set features
    - dir : str, directory to save plot
    - n_features_to_select : int, number of features to select
    - print_features : bool, whether to print feature scores or not (default: True)
    - df_n (int): number of the dataset

    Returns:
    - selected_feat_names : list of selected feature names
    """
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, f_classif)

    dict_name_score = dict(zip(fs.get_feature_names_out(), fs.scores_))
    max_scores = sorted(fs.scores_)[-n_features_to_select:]
    selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

    # Print all feature scores
    if print_features:
        for k,v in dict_name_score.items():
            print('Feature', k, ':', round(v,2))

    # plot the scores
    save_plot(X_train.columns, fs.scores_, f'anova_{df_n}.pdf', dir)
    return selected_feat_names





    



