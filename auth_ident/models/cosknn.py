from auth_ident.models import GenericSecondaryClassifier 
from auth_ident.datasets import ClosedDataset

from sklearn.model_selection import cross_val_score, train_test_split, KFold

import numpy as np
from scipy import spatial
from scipy.stats import mode
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors._base import NeighborsBase


class CosKNN(NeighborsBase):

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.train_x = X
        self.train_y = y

    def score(self, X, y):

        num_samples = X.shape[0]
        cos_dist = cosine_similarity(X, self.train_x)
        # Pull top k choices and reverse list to be closest....furthest
        top_k = cos_dist.argsort(axis=1)[:, -self.k:][:, ::-1]

        # Get labels of choices
        top_k_labels = self.train_y[top_k].reshape(num_samples, -1)
        prediction, _ = mode(top_k_labels, axis=1)

        score = np.sum(prediction.flatten() == y) / y.shape[0] 
        return score


class CosKNNSecondaryClassifier(GenericSecondaryClassifier):
    """
    CosKNNSecondaryClassifier
    """
    def __init__(self, params, combination, logger, logdir):
        super().__init__(params, combination, logger, logdir)

        self.name = "k_nearest_neighbor"
        self.dataset = ClosedDataset
        self.n_neighbors = params["model_params"]["n_neighbors"]

        self.model = CosKNN(self.n_neighbors)

    def train(self, X, y):

        # Cross val is probably randomizing data then spliting.
        # This is an issue because we want to split on author
        cv = KFold(n_splits=self.params["k_cross_val"], shuffle=False)

        results = cross_val_score(self.model,
                                  X,
                                  y,
                                  verbose=0,
                                  cv=cv)


        return {"accuracy": sum(results) / float(self.params["k_cross_val"])}

    def evaluate(self, X, y=None): 

        if y is None:
            return self.model.predict(X) 
        else:
            return self.model.score(X, y)

