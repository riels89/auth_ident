from auth_ident import GenericExecute
from auth_ident import TRAIN_LEN, VAL_LEN
from auth_ident.utils import accuracy
from auth_ident import param_mapping
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import os
import numpy as np


class TrainContrastive(GenericExecute):
    """
    Is the training class for the contrastive model.

    Uses the `contrastive` dictionary in json.
    """
    def execute_one(self, contrastive_params, combination, logger):

        curr_log_dir = os.path.join(self.logdir,
                                    "combination-" + str(combination))
        logger.info(f"Current log dir: {curr_log_dir}")

        model = param_mapping.map_model(contrastive_params)()

        training_dataset = param_mapping.map_dataset(
            model.dataset_type, contrastive_params,
            contrastive_params["train_data"])

        val_dataset = param_mapping.map_dataset(model.dataset_type,
                                                contrastive_params,
                                                contrastive_params["val_data"])

        param_mapping.map_params(contrastive_params)

        model = model.create_model(contrastive_params, combination, logger)

        tensorboard_callback = TensorBoard(log_dir=curr_log_dir,
                                           update_freq=64,
                                           profile_batch=0)

        save_model_callback = ModelCheckpoint(
            curr_log_dir +
            "/checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5",
            monitor='val_loss',
            save_best_only=True,
            mode='min')

        model.compile(optimizer=contrastive_params[combination]['optimizer'],
                      loss=contrastive_params[combination]['loss'],
                      metrics=[accuracy])

        model.summary()

        logger.info('Fit model on training data')

        history = model.fit(training_dataset,
                            validation_data=val_dataset,
                            epochs=contrastive_params[combination]['epochs'],
                            steps_per_epoch=TRAIN_LEN //
                            contrastive_params[combination]['batch_size'],
                            validation_steps=VAL_LEN //
                            contrastive_params[combination]['batch_size'],
                            callbacks=[
                                tensorboard_callback, save_model_callback
                            ])

        self.save_metrics(history.history, combination, curr_log_dir)

    def load_hyperparameter_matrix(self):

        hyperparameter_matrix_path = os.path.join(self.logdir,
                                                  "hyperparameter_matrix.csv")
        if os.path.isfile(hyperparameter_matrix_path):
            parameter_metrics = pd.read_csv(hyperparameter_matrix_path)
        else:
            parameter_metrics = None

        return parameter_metrics

    def save_metrics(self, history, combination, curr_log_dir):

        if self.parameter_metrics is None:
            self.parameter_metrics = pd.DataFrame(self.contrastive_params)
            self.parameter_metrics["val_accuracy"] = np.nan
            self.parameter_metrics["val_loss"] = np.nan

        self.parameter_metrics.loc[combination,
                                   'val_loss'] = history['val_loss'][0]
        self.parameter_metrics.loc[combination,
                                   'val_accuracy'] = history['val_accuracy'][0]


if __name__ == "__main__":
    TrainContrastive().execute()
