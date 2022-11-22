from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_selection.constants import PLOTFORMAT
from feature_selection.utils.main_utils import get_metrics, save_plot

class TreeBasedModels:

    def __init__(self, X_train, y_train, X_test, y_test, time_stamp, print_features=True, plot = True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test =  X_test
        self.y_test = y_test
        self.print_features = print_features
        self.plot = plot
        self.time_stamp = time_stamp
        
        

    def model(self, model_name):
        # define the model
        if model_name == "dtree":
            model = DecisionTreeClassifier()
        elif model_name == "rforest":
            model = RandomForestClassifier()
        else:
            model = XGBClassifier()

        # fit the model
        model.fit(self.X_train, self.y_train)
        # predict
        y_pred = model.predict(self.X_test)
        metrics_dict = get_metrics(self.y_test, y_pred)

        # Print metrics
        for k, v in metrics_dict.items():
            print(k, ":", v)

        # get importances
        importance = model.feature_importances_
        # Print all feature scores
        if self.print_features:
            for k,v in zip(self.X_train.columns, importance):
                print('Feature', k, ':', round(v,2))

        if self.plot:
            save_plot(self.X_train.columns, importance, model_name+PLOTFORMAT, self.time_stamp)

        return metrics_dict
