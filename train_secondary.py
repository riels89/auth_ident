from sklearn.model_selection import cross_val_score
from itertools import product
from tensorflow.keras import backend as K
from auth_ident import GenericExecute
from auth_ident import param_mapping
import os
import pandas as pd
from auth_ident.utils import get_embeddings, get_data, get_model
import time


class TrainSecondaryClassifier(GenericExecute):
    """
    Is the training class for the secondary classifiers.

    Uses a separate `secondary` dictionary in json, but still needs the 
    `contrastive` param dict to use the correct contrastive models and save the
    models to the correct folders.

    """
    def __init__(self):
        super().__init__()

    def execute_one(self, contrastive_params, combination, logger):
        """
        Loops over all of the secondary parameter combinations for each
        contrastive combination.
        """

        if self.mode == 'test':
            self.secondary_params = self.parameter_metrics.groupby(
                'combination')['accuracy'].max()
        param_mapping.map_params(contrastive_params)

        secondary_params_to_iterate = [
            self.secondary_params[comb] for comb in self.secondary_combs
        ]
        logger.info(
            f"secondary_params_to_iterate: {secondary_params_to_iterate}")

        curr_k_cross_val = None
        curr_max_authors = None
        for secondary_comb, params in enumerate(secondary_params_to_iterate):

            logger.info(f"secondary comb: {secondary_comb}, params: {params}")
            secondary_logdir = os.path.join(self.logdir,
                                            f"combination-{combination}",
                                            "validation",
                                            "secondary_classifier",
                                            f"combination-{secondary_comb}")
            os.makedirs(secondary_logdir, exist_ok=True)

            file_param = "val_data" if self.mode == "train" else "test_data"

            if "output_layer" in params["model_params"]:
                output_layer_name = params["model_params"]["output_layer"]
            else:
                output_layer_name = 'output_embedding'
            del params['model_params']['output_layer']

            self.model = param_mapping.map_model(params)(params,
                                                         combination,
                                                         logger,
                                                         self.logdir)
            params['model_params']['output_layer'] = output_layer_name

            print(f"max_authors {params['max_authors']}")
            print(f"curr_max_authors {curr_max_authors}")
            print(f"k_cross_val: {params['k_cross_val']}")
            print(f"curr k_cross_val: {curr_k_cross_val}")
            if self.model.name != "end_to_end_mlp" and (params['k_cross_val'] != curr_k_cross_val or curr_max_authors != params["max_authors"]):
                curr_k_cross_val = params['k_cross_val']
                curr_max_authors = params['max_authors']
                data_file = contrastive_params[file_param]

                train_data, train_labels = get_embeddings(
                    params=contrastive_params,
                    dataset=self.model.dataset,
                    max_authors=params["max_authors"],
                    k_cross_val=params['k_cross_val'],
                    output_layer_name=output_layer_name,
                    data_file=data_file,
                    combination=combination,
                    logger=logger,
                    logdir=self.logdir,
                    normalize= output_layer_name == "output_embedding")
                print("not end to end")

            elif self.model.name == "end_to_end_mlp":
                data_file = contrastive_params[file_param]
                curr_k_cross_val = params['k_cross_val']

                train_data, train_labels = get_data(contrastive_params, 
                                      self.model.dataset, 
                                      curr_k_cross_val,
                                      data_file)

                base_model = get_model(contrastive_params,
                                       output_layer_name,
                                       normalize=False,
                                       combination=combination,
                                       logger=logger,
                                       logdir=self.logdir)

                print("end to end")
                self.model.set_base_model(base_model)
                self.model.set_encoding_len(contrastive_params['dataset'].len_encoding)
            print(train_data.shape)

            results = self.model.train(train_data, train_labels)

            print(f"Results: {results}")
            self.model.save(secondary_logdir)

            self.save_metrics(results, params, combination)

        return results

    def load_hyperparameter_matrix(self):

        if self.mode == 'train':
            hyperparameter_matrix_path = os.path.join(
                self.logdir, "secondary_hyperparameter_matrix.csv")
        else:
            hyperparameter_matrix_path = os.path.join(
                self.logdir, "secondary_test_results_matrix.csv")
        if os.path.isfile(hyperparameter_matrix_path):
            parameter_metrics = pd.read_csv(
                hyperparameter_matrix_path,
                index_col=False).to_dict(orient='records')
        else:
            parameter_metrics = None
        print(f"Loaded hyper parameters: {parameter_metrics}")

        return parameter_metrics

    def save_metrics(self, results, params, combination):

        secondary_params = {"combination": combination, **params.copy()}
        del secondary_params['model_params']

        results_dict = {
            **secondary_params, "model_params": params['model_params'],
            **results
        }

        if self.parameter_metrics is None:
            self.parameter_metrics = [results_dict]
        else:
            self.parameter_metrics.append(results_dict)

        self.output_hyperparameter_metrics(self.logdir)

    def output_hyperparameter_metrics(self, directory):

        if self.mode == 'train':
            pd.DataFrame(self.parameter_metrics).to_csv(os.path.join(
                directory, "secondary_hyperparameter_matrix.csv"),
                                                        index=False)
        else:
            self.parameter_metrics.to_csv(
                os.path.join(directory, "secondary_test_results.csv"))

    def make_arg_parser(self):
        super().make_arg_parser()
        self.parser.add_argument("-mode")
        self.parser.add_argument("-second_combs", nargs='+', type=int)

    def get_args(self):

        exp_type, exp, combination = super().get_args()

        self.secondary_combs = self.args["second_combs"]
        self.mode = self.args["mode"]
        if self.mode is None:
            self.mode = "train"

        return exp_type, exp, combination


if __name__ == "__main__":
    TrainSecondaryClassifier().execute()
