from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from feature_selection.constants import PLOTFORMAT
from feature_selection.utils.main_utils import get_metrics, save_plot
from feature_selection.intrinsic.tree_based import TreeBasedModels

class CoefficientBasedModels(TreeBasedModels):

    def model(self, model_name):
        if model_name=="logreg":
            model = LogisticRegression()
        else:
            model= SVC(kernel = "linear")

        # fit the model
        model.fit(self.X_train, self.y_train)

        # predict
        y_pred = model.predict(self.X_test)
        metrics_dict = get_metrics(self.y_test, y_pred)

        # get importances
        importance = model.coef_[0]

        # Print all feature scores
        if self.print_features:
            for k,v in zip(self.X_train.columns, importance):
                print('Feature', k, ':', round(v,2))

        # plot feature importance
        if self.plot:
            save_plot(self.X_train.columns, importance, model_name+PLOTFORMAT, self.time_stamp)

        return metrics_dict
