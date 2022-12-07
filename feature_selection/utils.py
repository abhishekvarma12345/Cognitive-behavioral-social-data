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
import pandas as pd

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


def princ_comp_anal(X, dir, n_comp):
    pca = PCA().fit(X)
    fig = plt.figure(figsize=(15,10))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(os.path.join(dir, "pca.png"), dpi = 1200)
 
    principal = PCA(n_components=n_comp)
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


def compare_metrics(dict_full, dict_selected, model, show_metrics = True):
    
    new_dict = {}

    for key in dict_full.keys():
        new_dict[key] = [round(dict_full[key], 2), round(dict_selected[key], 2), round(100*(dict_selected[key]-dict_full[key])/dict_full[key], 2)] 

    df = pd.DataFrame.from_dict(new_dict)
    df["Features"] = ["All", "Selected", "Change (%)"]

    if show_metrics:
        print(df.set_index("Features")) 

    return df["Accuracy"].iloc[2]




def bar_plot(df, dir, filename):
    
    df_grouped = df.groupby(by=['CONDITION'], as_index = False).mean()

    df_to_plot = pd.DataFrame({
    'Columns': df_grouped.columns[1:],
    'H': df_grouped[df_grouped['CONDITION'] == 'H'].values.flatten().tolist()[1:],
    'D': df_grouped[df_grouped['CONDITION'] == 'D'].values.flatten().tolist()[1:]
    })

    fig, ax = plt.subplots(figsize=(12, 10))  
    ax = df_to_plot.plot.bar(x = 'Columns' , y = ['H','D'] , rot=0, xlabel = "Features")
    plt.savefig(os.path.join(dir, filename), dpi = 1200)



def how_many_common(selected_feat_list):

    l = selected_feat_list[0]

    for feat_list in selected_feat_list[1:]:
        l = list(set(l).intersection(feat_list))

    print(f"Out of {len(selected_feat_list[0])} selected features, the following {len(l)} are common to all methods: {l}")




def model_accuracy_comparison(X_train_scaled, X_test_scaled, y_train_encoded,
                              y_test_encoded, mydir, dtree_metrics, rforest_metrics,
                              xgboost_metrics, logreg_metrics, svm_metrics, selector,
                              models, n_features_list, n_features_to_select):
    
    acc_tree, acc_for, acc_xgb, acc_lr, acc_svm = [], [], [], [], []

    for i in n_features_list:

        selected_features = selector(X_train_scaled, y_train_encoded, X_test_scaled, mydir, i, print_features = False)
        X_train_reduced = X_train_scaled.loc[:,selected_features]
        X_test_reduced = X_test_scaled.loc[:,selected_features]

        dtree_metrics_red = models[0](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        rforest_metrics_red = models[1](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        xgboost_metrics_red = models[2](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        logreg_metrics_red = models[3](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        svm_metrics_red = models[4](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)

        acc_tree.append(compare_metrics(dtree_metrics, dtree_metrics_red, "Decision tree", show_metrics = False))
        acc_for.append(compare_metrics(rforest_metrics, rforest_metrics_red, "Random Forest", show_metrics = False))
        acc_xgb.append(compare_metrics(xgboost_metrics, xgboost_metrics_red, "XGBoost", show_metrics = False))
        acc_lr.append(compare_metrics(logreg_metrics, logreg_metrics_red, "Logistic regression", show_metrics = False))
        acc_svm.append(compare_metrics(svm_metrics, svm_metrics_red, "Support vector machine", show_metrics = False))



    name_plot = str(selector).split(' ')[1] + "_behavior_number_features.png"
    fig = plt.figure(figsize=(12,8))
    plt.plot(n_features_list, acc_tree, label = "Decision tree", marker='.')
    plt.plot(n_features_list, acc_for, label = "Random forest", marker='.')
    plt.plot(n_features_list, acc_xgb, label = "XGBoost", marker='.')
    plt.plot(n_features_list, acc_lr, label = "Logistic regression", marker='.')
    plt.plot(n_features_list, acc_svm, label = "SVM", marker='.')
    plt.axvline(x = n_features_to_select, color = 'k', linestyle='dashdot', alpha = 0.5)
    plt.xlabel("Number of selected features")
    plt.ylabel("Change in accuracy (%)")
    plt.legend(prop={'size': 14})
    plt.savefig(os.path.join(mydir, name_plot), dpi = 1200)



def plot_heatmap(models, selection_methods, matrix, mydir):

    fig, ax = plt.subplots(figsize=(12,8))
    im = ax.imshow(matrix, cmap = 'Reds')
    labels_selectors = [str(i).split(' ')[1] for i in selection_methods]
    labels_models = [str(i).split(' ')[1] for i in models]

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(models)), labels=labels_models)
    ax.set_yticks(np.arange(len(selection_methods)), labels=labels_selectors)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(selection_methods)):
        for j in range(len(models)):
            text = ax.text(j, i, matrix[i, j],
                        ha="center", va="center", color="r", size = 18)

    ax.set_title("Change in accuracy (%)")
    fig.tight_layout()
    plt.savefig(os.path.join(mydir, "heatmap_accuracy"), dpi = 1200)



def heatmap(X_train_scaled, X_test_scaled, y_train_encoded, 
            y_test_encoded, mydir, dtree_metrics, rforest_metrics, 
            xgboost_metrics, logreg_metrics, svm_metrics, 
            selection_methods, models, selected_features_list):


    # models = ['Decision tree', 'Random forest', 'XGBoost', 'Logistic regression', 'SVM']
    matrix = np.zeros((len(selection_methods), len(models)))

    for i in range(len(selection_methods)):

        X_train_reduced = X_train_scaled.loc[:, selected_features_list[i]]
        X_test_reduced = X_test_scaled.loc[:, selected_features_list[i]]

        dtree_metrics_red = models[0](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        rforest_metrics_red = models[1](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        xgboost_metrics_red = models[2](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        logreg_metrics_red = models[3](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
        svm_metrics_red = models[4](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)

        matrix[i][0] = compare_metrics(dtree_metrics, dtree_metrics_red, "Decision tree", show_metrics = False)
        matrix[i][1]= compare_metrics(rforest_metrics, rforest_metrics_red, "Random Forest", show_metrics = False)
        matrix[i][2]= compare_metrics(xgboost_metrics, xgboost_metrics_red, "XGBoost", show_metrics = False)
        matrix[i][3]= compare_metrics(logreg_metrics, logreg_metrics_red, "Logistic regression", show_metrics = False)
        matrix[i][4]= compare_metrics(svm_metrics, svm_metrics_red, "Support vector machine", show_metrics = False)

    # plot the heatmap
    plot_heatmap(models, selection_methods, matrix, mydir)








def mean_change_accuracy(X_train_scaled, X_test_scaled, y_train_encoded, 
        y_test_encoded, mydir, dtree_metrics, rforest_metrics, 
        xgboost_metrics, logreg_metrics, svm_metrics, 
        selection_methods, models, n_features_list, n_features_to_select):

    my_list_3 = []
    for method in selection_methods:

        my_list_2 = []
        # acc_chi2, acc_mutinf, acc_anova, acc_perm = [], [], [], []

        for i in n_features_list:   

            selected_features = method(X_train_scaled, y_train_encoded, X_test_scaled, mydir, i, print_features = False)

            X_train_reduced = X_train_scaled.loc[:, selected_features]
            X_test_reduced = X_test_scaled.loc[:, selected_features]

            dtree_metrics_red = models[0](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
            rforest_metrics_red = models[1](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
            xgboost_metrics_red = models[2](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
            logreg_metrics_red = models[3](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)
            svm_metrics_red = models[4](X_train_reduced, y_train_encoded, X_test_reduced, y_test_encoded, mydir, print_features = False, plot = False)

            my_list = [
            compare_metrics(dtree_metrics, dtree_metrics_red, "Decision tree", show_metrics = False),
            compare_metrics(rforest_metrics, rforest_metrics_red, "Random Forest", show_metrics = False),
            compare_metrics(xgboost_metrics, xgboost_metrics_red, "XGBoost", show_metrics = False),
            compare_metrics(logreg_metrics, logreg_metrics_red, "Logistic regression", show_metrics = False),
            compare_metrics(svm_metrics, svm_metrics_red, "Support vector machine", show_metrics = False)
            ]

            avg = sum(my_list)/len(my_list)
            my_list_2.append(avg)

        my_list_3.append(my_list_2)


    fig = plt.figure(figsize=(12,8))
    plt.plot(n_features_list, my_list_3[0], label = "Chi2", marker='.')
    plt.plot(n_features_list, my_list_3[1], label = "Mutual information", marker='.')
    plt.plot(n_features_list, my_list_3[2], label = "Anova", marker='.')
    plt.plot(n_features_list, my_list_3[3], label = "Permutation importance", marker='.')
    plt.axvline(x = n_features_to_select, color = 'k', linestyle='dashdot', alpha = 0.5)
    plt.xlabel("Number of selected features")
    plt.ylabel("Mean change in accuracy (%)")
    plt.legend(prop={'size': 16})
    plt.savefig(os.path.join(mydir, "mean_change_in_accuracy.png"), dpi = 1200)

