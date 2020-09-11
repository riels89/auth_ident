class GenericSecondaryClassifier():
    def __init__(self, params, combination, logger):
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
        """

        self.params = params
        self.combination = combination
        self.logger = logger

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
