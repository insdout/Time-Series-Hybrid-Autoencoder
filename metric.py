from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


class KNNRULmetric:
    """
    Metric for evaluating the distribution of the true RUL values over latent space, given datapoint x.
    The datapoint x is mapped to latent space Z, for each vector in Z n nearest neighbours classifier is predicting
    a class {0, 1}. Where class 1 stands for z vectors with true RUL greater than rul_threshold
    and class 0 stands for vectors with true RUL less or equal to rul_threshold.

    The metric is calculated as ratio of misclassified vectors to total number of vectors.
    """
    def __init__(self, rul_threshold, n_neighbors):
        self.fitted = False
        self.rul_threshold = rul_threshold
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def fit(self, z, rul):
        """
        Fits KNeighbors Classifier.
        :param z: latent space vector representations, np.array
        :param rul: true RUL of the given latent space vector representations
        """
        # Converts RUL values to binary vector,
        # where rul_binary is 1 if RUL is greater than rul_threshold and 0 otherwise
        rul_binary = np.where(rul > self.rul_threshold, 1, 0)
        self.classifier = self.classifier.fit(z, rul_binary)
        self.fitted = True
    
    def calculate(self, z, rul):
        """
        Calculates the metric value as a count of misclassified vectors divided by total count of vectors.
        :param z: latent space vector representations, np.array
        :param rul: true RUL of the given latent space vector representations
        :return: metric value, int
        """
        assert self.fitted == True, "You should call fit first."
        if rul.max() > 1:
            rul_binary = np.where(rul >= self.rul_threshold, 1, 0)
        else:
            rul_binary = rul
        num_points = rul_binary.shape[0]
        healthy_pred = self.classifier.predict(z)
        sum_errors = np.sum(np.abs(healthy_pred - rul_binary))
        metric = sum_errors/num_points
        return metric
    
    def fit_calculate(self, z, rul):
        """
        Fits the KNeighbors Classifier and calculates the metric.
        :param z: latent space vector representations, np.array
        :param rul: true RUL of the given latent space vector representations
        :return: metric value, int
        """
        self.fit(z, rul)
        return self.calculate(z, rul)
    
    def infer_point(self, z):
        """
        Checks that classifier is fitted and infer the class for given vector z
        :param z: latent space vector representation, np.array
        :return: metric value, int
        """
        assert self.fitted == True, "You should call fit first."
        healthy_pred = self.classifier.predict(z)
        return healthy_pred
    
    def plot_zspace(self, z, rul , save=False, show=False, path="", title=""):
        """

        :param z: latent space vector representations, np.array
        :param rul: true RUL of the given latent space vector representations
        :param save: whether to save the plot or not
        :param show: whether to show the plot or not
        :param path: path to the directory where plots will be saved, string
        :param title: Title of the plot, str
        """
        assert self.fitted == True, "You should call fit first."
        if rul.max() > 1:
            rul_binary = np.where(rul >= self.rul_threshold, 1, 0)
        else:
            rul_binary = rul
        num_points = rul_binary.shape[0]
        healthy_pred = self.classifier.predict(z)
        sum_errors = np.sum(np.abs(healthy_pred - rul_binary))
        metric = sum_errors/num_points

        type_errors = rul_binary - healthy_pred

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(10, 5))
    
        pa = ax[0].scatter(z[:, 0], z[:, 1], c=rul, s=3, alpha=1)
        divider = make_axes_locatable(ax[0])
        ccx = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(pa, cax=ccx)
        ax[0].set_title("Cluster Assignments")
        ax[0].set_xlabel("z - dim 1")
        ax[0].set_ylabel("z - dim 2")
        
        colors = np.array(["green", "red", "orange"])
        alphas = [1, 1, 1]
        sizes = [3, 7, 7]
        for i, error_type in enumerate([0,-1, 1]):
            mask = (type_errors == error_type)
            ax[1].scatter(z[mask, 0], z[mask, 1], c=colors[type_errors[mask]], s=sizes[i], alpha=alphas[i])
        ax[1].set_title(f"Errors. Metric: {np.round(metric, 5)}")
        ax[1].set_xlabel("z - dim 1")
        ax[1].set_ylabel("z - dim 2")

        if save:
            img_path = path+'/images/'
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path +'metric_'+str(title)+'.png')
        if show:
            plt.show()
