import os, datetime
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
import scipy.stats as ss
import numpy as np
from dython.nominal import conditional_entropy
from dython.nominal import Counter
import seaborn as sns
from sklearn.decomposition import PCA

def make_timestamp_dir(folder_name):
    mydir = os.path.join(os.getcwd(), 'feature_selection', 'artifacts', folder_name,
                         datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if os.path.exists(mydir) == False:
        os.makedirs(mydir)
        
    return mydir

def save_plot(columns, feature_importances, filename, dir):
    fig = plt.figure(figsize=(15,10))
    plt.bar(columns, feature_importances)
    plt.xlabel("features")
    plt.ylabel("feature importance")
    plt.savefig(os.path.join(dir, filename))
    return fig

#def merge_plots(plot_1, plot_2, plot_3, filename):
#    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (25,20))
#    ax[0,0] = plot_1
#    ax[0,1] = plot_2
#    ax[1,0] = plot_3
#    filename = "testing"
#    plt.savefig(os.path.join(os.getcwd(), 'feature_selection', 'artifacts', filename))

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

def save_plot_sns(corr, filename, folder_name):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = sns.heatmap(corr, annot=True, ax=ax)
    plt.savefig(os.path.join(os.getcwd(), 'feature_selection', 'artifacts', folder_name, filename))


def princ_comp_anal(X, folder_name):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(os.path.join(os.getcwd(), 'feature_selection', 'artifacts', folder_name, "pca.png"))
 
    principal = PCA(n_components=2)
    principal.fit(X)
    X_pca = principal.transform(X)
    
    # Check the dimensions of data after PCA
    print("New dimension of data: ", X_pca.shape)

    return X_pca
