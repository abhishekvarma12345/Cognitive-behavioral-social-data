# filter methods

# considering input datatype:ordinal, output datatype:categorical
from sklearn.feature_selection import chi2, mutual_info_classif

# considering input datatype:numerical, output datatype:categorical
from sklearn.feature_selection import f_classif

from feature_selection.constants import PLOTFORMAT
from feature_selection.utils.main_utils import save_plot, select_features

class StatisticBasedModels:
    def __init__(self, X_train, y_train, X_test, y_test, time_stamp, n_features_to_select, print_features=True, plot = True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test =  X_test
        self.y_test = y_test
        self.time_stamp = time_stamp
        self.n_features_to_select = n_features_to_select
        self.print_features = print_features
        self.plot = plot
        
        
    def stat_model(self,model_name):
        
        # feature selection
        if model_name == "chi_2":
            fs = select_features(self.X_train, self.y_train, self.X_test, chi2)
        elif model_name == "mutual_inf":
            fs = select_features(self.X_train, self.y_train, self.X_test, mutual_info_classif)
        else:
            fs = select_features(self.X_train, self.y_train, self.X_test, f_classif)

        dict_name_score = dict(zip(fs.get_feature_names_out(), fs.scores_))
        max_scores = sorted(fs.scores_)[-self.n_features_to_select:]
        selected_feat_names = [k for k, v in dict_name_score.items() if v in max_scores]

        # Print all feature scores
        if self.print_features:
            for k,v in dict_name_score.items():
                print('Feature', k, ':', round(v,2))

        # plot the scores
        if self.plot:
            save_plot(self.X_train.columns, fs.scores_, model_name+PLOTFORMAT, self.time_stamp)
        return selected_feat_names


