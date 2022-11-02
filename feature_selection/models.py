from feature_selection.utils import save_plot,select_features , cramers_corrected_stat, theils_u, save_plot_sns


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import numpy as np
from itertools import combinations


def dtree(X, y, X_test, y_test, folder_name, action=None):
    # define the model
    model = DecisionTreeClassifier()
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    # Print metrics
    print("Decision tree metrics: ")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Roc_auc:", metrics.roc_auc_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    print("Feature importance: ")
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    dtree_plot = save_plot(X.columns, importance, 'dtree.png', folder_name)
    return dtree_plot

def rforest(X, y, X_test, y_test, folder_name, action=None):

    model = RandomForestClassifier()
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    # Print metrics
    print("Random forest metrics: ")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Roc_auc:", metrics.roc_auc_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    print("Feature importance: ")
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    rforest_plot = save_plot(X.columns, importance, 'rforest.png', folder_name)
    return rforest_plot

def xgboost(X, y, X_test, y_test, folder_name, action=None):
    # define the model
    model = XGBClassifier()
    # fit the model
    model.fit(X, y)
    # predict
    y_pred = model.predict(X_test)
    # Print metrics
    print("Random forest metrics: ")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Roc_auc:", metrics.roc_auc_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    print("Feature importance: ")
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    xgboost_plot = save_plot(X.columns, importance, 'xgboost.png', folder_name)
    return xgboost_plot


def perm_knn(X,y, folder_name, action = None):
    # define the model
    model = KNeighborsClassifier()
    # fit the model
    model.fit(X, y)
    # perform permutation importance
    results = permutation_importance(model, X, y, scoring='accuracy')
    # get importance
    importance = results.importances_mean
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    perm_plot = save_plot(X.columns, importance, 'perm_knn.png', folder_name)
    return perm_plot 

 
def chi_2(X_train, y_train, X_test, folder_name):
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, chi2)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    chi2_plot = save_plot(X_train.columns, fs.scores_, 'chi_2.png', folder_name)


def mutual_inf(X_train, y_train, X_test, folder_name):
    # feature selection
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test, mutual_info_classif)
    # what are scores for the features
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))
    # plot the scores
    mutualinf_plot = save_plot(X_train.columns, fs.scores_, 'mutual_inf.png', folder_name)


def categorical_corr(df, folder_name):
    cols = df.columns
    corrM = np.zeros((len(cols),len(cols)))
    for col1, col2 in combinations(cols, 2):
        idx1, idx2 = cols.get_loc(col1), cols.get_loc(col2)
        corrM[idx1, idx2] = cramers_corrected_stat(pd.crosstab(df[col1], df[col2]))
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    save_plot_sns(corr, 'categorical_correlation.png', folder_name)


def unc_coeff(df, folder_name):
    cols = df.columns
    corrM = np.zeros((len(cols),len(cols)))
    for col1, col2 in combinations(cols, 2):
        idx1, idx2 = cols.get_loc(col1), cols.get_loc(col2)
        corrM[idx1, idx2] = theils_u(df[col1], df[col2])
        corrM[idx2, idx1] = corrM[idx1, idx2]

    corr = pd.DataFrame(corrM, index=cols, columns=cols)
    save_plot_sns(corr, 'uncertainty_coefficients.png', folder_name)

    



