import os
import matplotlib.pyplot as plt



def save_plot(columns, feature_importances, filename):
    plt.figure(figsize=(20,10))
    plt.bar(columns, feature_importances)
    plt.xlabel("features")
    plt.ylabel("feature importance")
    plt.savefig(os.path.join(os.getcwd(), 'feature_selection', 'artifacts',filename))
    