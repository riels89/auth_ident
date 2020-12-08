import os
import pickle


class GenericSecondaryClassifier():
    def __init__(self, params=None, combination=None, logger=None, logdir=None):
        """
        Creates a new SecondaryClassifier:

        Args:

            params (`dict`):

                Contains the secondary parameters. Puts secondary params
                "model_params", directly into the SK-Learn model as kwargs.

            combination (`int`):

                Current contrastive combination.

            logger (`python logger`):

                Logger for this model

            logdir ('str'):
                
                Logging directory
        """
        assert logger is not None, "must give logger param"

        self.params = params
        self.combination = combination
        self.logger = logger
        self.logdir = logdir

        self.name = None
        self.dataset = None
        self.model = None

    def train(self, X, y):
        """
        Trains the model with the given data.
        Puts secondary params "model_params", directly into the SK-Learn
        model as kwargs.

        Args:

            X (`numpy array`):
                
                The training data

            y (`numpy array`):
                
                The labels
        """
        pass

    def evaluate(self, X, y=None):
        """
        Evaluates the model with the given data.
        If given labels, the score will be returned.

        Args:

            X (`numpy array`):
                
                The data to evaluate.

            y (`numpy array, optional`):

                If given, the data will be evaluated AND the score will be
                retuend with the given labels.
                
                
        """
        pass

    def save(self, secondary_logdir):

        with open(os.path.join(secondary_logdir, "secondary_classifier.pkl"),
                  'wb') as f:
            to_save = {"model": self.model, 
                       "params": self.params,
                       "combination": self.combination}
            pickle.dump(to_save, f)

    def load(self, secondary_logdir):

        model_path = [
            os.path.join(secondary_logdir, f) for f in os.listdir(secondary_logdir)
            if f.endswith(".pkl")
        ]
        model_path = max(model_path, key=os.path.getctime)
        data = pickle.loads(model_path)

        self.model = data["model"]
        self.params = data["params"]
        self.combination = data["combination"]

