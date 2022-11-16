import os, datetime
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
import scipy.stats as ss
import numpy as np
from dython.nominal import conditional_entropy
from dython.nominal import Counter
import seaborn as sns
import matplotlib.image as mpimg
from sklearn.decomposition import PCA
from sklearn import metrics

def make_timestamp_dir(folder_name):
    mydir = os.path.join(os.getcwd(), 'feature_selection', 'artifacts', folder_name,
                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(mydir, exist_ok=True)

    return mydir

def save_plot(columns, feature_importances, filename, dir):
    fig = plt.figure(figsize=(15,10))
    plt.bar(columns, feature_importances)
    plt.xlabel("features")
    plt.ylabel("feature importance")
    plt.savefig(os.path.join(dir, filename))
    return fig

def merge_plots(plots_dir, filename):
    img1 = mpimg.imread(os.path.join(plots_dir, 'dtree.png'))
    img2 = mpimg.imread(os.path.join(plots_dir, 'rforest.png'))
    img3 = mpimg.imread(os.path.join(plots_dir, 'xgboost.png'))
    img4 = mpimg.imread(os.path.join(plots_dir, 'perm_knn.png'))
    img5 = mpimg.imread(os.path.join(plots_dir, 'chi_2.png'))    
    img6 = mpimg.imread(os.path.join(plots_dir, 'mutual_inf.png'))
    # img7 = mpimg.imread(os.path.join(plots_dir, 'uncertainty_coefficients.png'))
    # img8 = mpimg.imread(os.path.join(plots_dir, 'categorical_correlation.png'))
    
    fig , ax = plt.subplots(2, 3, figsize=(15, 7), constrained_layout = True)
    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.01, hspace=0.01)
    ax[0,0].imshow(img1)
    ax[0,0].axis("off")
    ax[0,0].set_title('Decision tree', y = 1.0, pad = -15)
    ax[0,1].imshow(img2)
    ax[0,1].axis("off")
    ax[0,1].set_title('Random forest', y = 1.0, pad = -15)
    ax[0,2].imshow(img3)
    ax[0,2].axis("off")
    ax[0,2].set_title('XGBoost', y = 1.0, pad = -15)
    ax[1,0].imshow(img4)
    ax[1,0].axis("off")
    ax[1,0].set_title('Permutation with knn', y = 1.0, pad = -15)
    ax[1,1].imshow(img5)
    ax[1,1].axis("off")
    ax[1,1].set_title(r'$\chi^{2}$', y = 1.0, pad = -15)
    ax[1,2].imshow(img6)
    ax[1,2].axis("off")
    ax[1,2].set_title('Mutual information', y = 1.0, pad = -15)
    # ax[1,2].imshow(img7)
    # ax[1,2].axis("off")
    # ax[1,3].imshow(img8)
    # ax[1,3].axis("off")
    fig.savefig(os.path.join(plots_dir, filename), dpi=1200)

def select_features(X_train, y_train, X_test, score_f):
	fs = SelectKBest(score_func=score_f, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorical-categorical association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
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
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x

def save_plot_sns(corr, filename, dir):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = sns.heatmap(corr, annot=True, ax=ax)
    plt.savefig(os.path.join(dir, filename))


def princ_comp_anal(X, dir):
    pca = PCA().fit(X)
    fig = plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(os.path.join(dir, "pca.png"))
 
    principal = PCA(n_components=2)
    principal.fit(X)
    X_pca = principal.transform(X)
    
    # Check the dimensions of data after PCA
    print("New dimension of data: ", X_pca.shape)

    return X_pca


def get_metrics(y_test, y_pred):
    metrics_dict = {"Accuracy" : metrics.accuracy_score(y_test, y_pred),
                    "Roc_auc" : metrics.roc_auc_score(y_test, y_pred),
                    "F1" : metrics.f1_score(y_test, y_pred),
                    "Precision" : metrics.precision_score(y_test, y_pred),
                    "Recall" : metrics.recall_score(y_test, y_pred)}

    return metrics_dict


def compare_metrics(dict_full, dict_selected, model):
    print(model.center(50,'*'))
    print("Metric ", "All features ", "Selected Features ")
    for key in dict_full.keys():
        print(key, round(dict_full[key], 2), round(dict_selected[key], 2))
