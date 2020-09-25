import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import logging
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import backend as K
from shutil import copy
from auth_ident import param_mapping
import argparse
import json
from pprint import pprint


class GenericExecute:
    """
    This class contains the shared logic between the different 
    training/validation or other tasks we need to do.

    Some of the shared logic includes:

    * The argparser
    * Geting getting the correct directory for each expirement and combination
    * Loads param dict and creates neccessary combinations
    * The primary model loop
    * Creating the logger and some basic logging 
    * Set up a structure for loading, updating, and saving hyperparameter 
    results

    This class is set up in a way to easily overwrite each method for custom
    functions which can varry widley.
    """
    def __init__(self):

        self.make_arg_parser()
        self.exp_type, self.exp_num, self.combinations = self.get_args()

        self.logdir, exp_dir = self.get_logdir(self.exp_type, self.exp_num)

        params = json.load(open(os.path.join(self.logdir, "param_dict.json")))

        self.root_logger = self.create_logger(self.logdir, name='root')
        self.root_handler = self.root_logger.handlers[0]
        self.root_logger.info(
            f"Training with combinations: {self.combinations}")

        self.contrastive_params = param_mapping.generate_param_grid(
            params['contrastive'])
        non_model_secondary_params = param_mapping.generate_param_grid(
            params['secondary'])

        if -1 in self.combinations:
            self.combinations = list(range(len(self.contrastive_params)))

        # Get param combs for each model's params
        self.secondary_params = []
        num_models = len(params['secondary']['model'])
        combs_per_model = int(len(non_model_secondary_params) / num_models)
        for i in range(num_models):
            secondary_classifier_params = param_mapping.generate_param_grid(
                params['secondary']["model_params"][0][i])
            for curr_grid in range(i * combs_per_model,
                                   (i + 1) * combs_per_model):
                for model_param_set in secondary_classifier_params:
                    curr_params = non_model_secondary_params[curr_grid].copy()
                    curr_params['model_params'] = model_param_set
                    self.secondary_params.append(curr_params)

        pprint(self.secondary_params)

        self.parameter_metrics = self.load_hyperparameter_matrix()

    def execute(self):
        """
        Handles the primary loop logic, calls the function execute_one each
        iteration and is where the main logic should be for downstream tasks.
        """

        expirement_name = self.logdir.split('/')[1:2]
        self.root_logger.info(f"Starting expirement {expirement_name}")
        self.root_logger.info("")

        for combination in self.combinations:

            curr_log_dir = os.path.join(self.logdir,
                                        "combination-" + str(combination))
            os.makedirs(curr_log_dir, exist_ok=True)
            assert os.path.isdir(
                curr_log_dir), "Dir " + curr_log_dir + "doesn't exist"
            os.makedirs(curr_log_dir + '/checkpoints', exist_ok=True)
            assert os.path.isdir(
                curr_log_dir +
                '/checkpoints'), "Dir " + curr_log_dir + "doesn't exist"

            logger = self.create_logger(curr_log_dir, index=combination)
            logger.addHandler(self.root_handler)

            logger.info('Training with parameter combination ' +
                        str(combination))
            logger.info(
                f"With parameters: {str(self.contrastive_params[combination])}"
            )
            logger.info("")

            self.execute_one(self.contrastive_params[combination], combination,
                             logger)

        self.output_hypeparameter_metrics(self.logdir)

    def execute_one(self, contrastive_params, combination, logger):
        """
        Used by execute and will be called for each iteration of the primary 
        loop.

        Args: 
            contrastive_params (`dict`):

                Contains the parameters for the current contrastive combination
                
            combination (`int`):
                
                Current contrastive combination.

            logger (`python logger`):

                The logger for the current contrastive model.
        """
        pass

    def load_hyperparameter_matrix(self):

        return None

    def output_hypeparameter_metrics(self, directory):
        self.parameter_metrics.to_csv(
            os.path.join(directory, "hyperparameter_matrix.csv"))

    def save_metrics(self, history, combination, curr_log_dir):
        pass

    def create_logger(self, log_dir, name=None, index=None):
        if name is None:
            assert index is not None, "Must specify index in create logger"
            name = 'combination-' + str(index)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_dir + '/' + name + ".log")
        formatter = logging.Formatter(
            '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def make_arg_parser(self):
        """
        Makes the argument argparser, should overwrite to add additional
        parameters.
        """
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('-type')
        self.parser.add_argument('-exp')
        self.parser.add_argument('-combinations', nargs='+')

    def get_args(self):
        """
        Actually gets the arguments from the argparser. Calls 
        self.make_arg_parser to create the parser.
        """

        self.args = vars(self.parser.parse_args())

        exp = self.args['exp']

        exp_type = self.args['type']

        combination = list(map(int, self.args['combinations']))

        return exp_type, exp, combination

    def get_logdir(self, exp_type, expirement_num):
        """
        Will create the logdir and expirement_dir for the specified type
        and expirement_num.

        Organizing by types istead of by 'models' allows for more flexibility.
        Types can be anything.
        """

        expirement_dir = [
            filename
            for filename in os.listdir(os.path.join("models", exp_type))
            if filename.startswith('EXP' + str(expirement_num))
        ][0]

        logdir = os.path.join("models", exp_type, expirement_dir)

        assert os.path.isdir(logdir), "Dir " + logdir + " doesn't exist"

        return logdir, expirement_dir
