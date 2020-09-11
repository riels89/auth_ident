from sklearn.model_selection import cross_val_score
from itertools import product
from tensorflow.keras import backend as K
from auth_ident import GenericExecute
from auth_ident import param_mapping
import os
import tensorflow.keras as keras
import tensorflow as tf
import pickle
import pandas as pd


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

        for params in self.secondary_params:

            secondary_logdir = os.path.join(self.logdir,
                                            "secondary_classifier",
                                            "combination-" + str(combination))
            self.model = param_mapping.map_model(params)(params, combination,
                                                         logger)

            train_data, train_labels = self.get_embeddings(
                contrastive_params, self.model.dataset, params['k_cross_val'],
                combination, logger)

            results = self.model.train(train_data, train_labels)

            with open(os.path.join(secondary_logdir, self.model.name + ".pkl"),
                      'wb') as f:
                pickle.dump(self.model, f)

            self.save_metrics(results, params, combination)

            return results

    def load_hyperparameter_matrix(self):

        if self.mode == 'train':
            hyperparameter_matrix_path = os.path.join(
                self.logdir, "secondary_train_results_matrix.csv")
        else:
            hyperparameter_matrix_path = os.path.join(
                self.logdir, "secondary_test_results_matrix.csv")
        if os.path.isfile(hyperparameter_matrix_path):
            parameter_metrics = pd.read_csv(hyperparameter_matrix_path)
        else:
            parameter_metrics = None

        return parameter_metrics

    def save_metrics(self, results, params, combination):

        model_params = params['model_params'].copy()
        model_params['combination'] = combination
        results_dict = model_params + results

        if self.parameter_metrics is None:
            self.parameter_metrics = pd.DataFrame(results_dict)

        self.parameter_metrics.append(results_dict)

    def output_hypeparameter_metrics(self, directory):

        self.parameter_metrics.to_csv(
            os.path.join(directory, "secondary_hyperparameter_matrix.csv"))

    def get_embeddings(self, params, dataset, k_cross_val, combination,
                       logger):
        file_param = "val_data" if self.mode == "train" else "test_data"

        dataset = dataset(crop_length=params["max_code_length"],
                          k_cross_val=k_cross_val,
                          data_file=params[file_param])
        params['dataset'] = dataset
        data, labels = dataset.get_dataset()

        contrastive_model = param_mapping.map_model(params)()
        encoder = self.load_encoder(contrastive_model, params, combination,
                                    logger)

        layer_name = 'output_embedding'
        embedding_layer = keras.Model(
            inputs=encoder.input, outputs=encoder.get_layer(layer_name).output)
        embedding_layer.summary()

        embeddings = embedding_layer.evaluate(data,
                                              batch_size=params["batch_size"])

        return embeddings, labels

    def load_encoder(self, model, params, combination, logger):

        # Create inner model
        encoder = model.create_model(params, combination, self.root_logger)

        # Load most recent checkpoint
        checkpoint_dir = os.path.join(self.logdir, "/checkpoints")
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        encoder.load_weights(latest)

        return encoder

    def make_arg_parser(self):
        super().make_arg_parser()
        self.parser.add_argument("-mode")

    def get_args(self):

        exp_type, exp, combination = super().get_args()

        self.mode = self.args["mode"]

        return exp_type, exp, combination


if __name__ == "__main__":
    TrainSecondaryClassifier().execute()
