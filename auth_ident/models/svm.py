from auth_ident.models import GenericSecondaryClassifier 
from auth_ident.datasets import ClosedDataset

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split, KFold


class SVMSecondaryClassifier(GenericSecondaryClassifier):
    """
    SVMSecondaryClassifier

    Sk-learn SGDClassifier wrapper.
    """
    def __init__(self, params, combination, logger, logdir):
        super().__init__(params, combination, logger, logdir)

        self.name = "svm"
        self.dataset = ClosedDataset

        self.model = SVC(
            **self.params["model_params"])

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
